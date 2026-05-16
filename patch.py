import re
with open("scripts/train.py", "r") as f:
    content = f.read()
old_code = """    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # 멀티프로세싱 워커가 여러개일 경우 각자 다른 파일 위치부터 읽게 처리하면 좋지만,
        # 여기서는 단순화를 위해 워커가 하나라고 가정하거나 랜덤하게 넘기도록 처리
        buf = []
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            if worker_info is not None:
                # 워커마다 다른 부분을 읽도록 파일을 분할합니다.
                # 총 라인수를 알기 어려우므로 바이트 오프셋 기준으로 크게 뜁니다.
                # 예: 최대 25GB 파일이면 워커/노드별로 충분히 멀리 띄워줍니다.
                import os
                file_size = os.path.getsize(self.path)
                worker_id = worker_info.id
                num_workers = worker_info.num_workers
                # 시작 오프셋 계산
                chunk_size = file_size // num_workers
                start_offset = worker_id * chunk_size
                if start_offset > 0:
                    f.seek(start_offset)
                    # 첫 줄은 잘렸을 확률이 높으니 버립니다.
                    f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                buf.extend(self.tokenizer.encode(line, add_bos=False, add_eos=False))
                buf.append(self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 2)
                while len(buf) > self.seq_len * 4: # 약간 여유있게 버퍼링
                    x = torch.tensor(buf[:self.seq_len], dtype=torch.long)
                    y = torch.tensor(buf[1:self.seq_len+1], dtype=torch.long)
                    buf = buf[self.seq_len:]
                    yield x, y"""
new_code = """    def __iter__(self):
        import os
        import random
        import torch.distributed as dist
        worker_info = torch.utils.data.get_worker_info()
        buf = []
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            file_size = os.path.getsize(self.path)
            # --- 1. Calculate precise chunk for each worker considering DDP ---
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
            num_workers = worker_info.num_workers if worker_info is not None else 1
            worker_id = worker_info.id if worker_info is not None else 0
            total_workers = world_size * num_workers
            global_worker_id = rank * num_workers + worker_id
            chunk_size = file_size // total_workers
            start_offset = global_worker_id * chunk_size
            end_offset = start_offset + chunk_size if global_worker_id < total_workers - 1 else file_size
            if start_offset > 0:
                f.seek(start_offset)
                f.readline()  # dump fractional line
            # --- 2. Shuffle window initialization ---
            # Yielding lines in exact sequential order hurts training stability.
            # We buffer large numbers of tokens and pick randomly to simulate shuffling.
            shuffle_buffer = []
            max_shuffle_items = 1000  # Number of sequences to keep in shuffle cache
            def yield_from_buffer(force_all=False):
                nonlocal buf
                # Token buffer might not be full, try to slice pairs
                while len(buf) > self.seq_len * 2:
                    x = torch.tensor(buf[:self.seq_len], dtype=torch.long)
                    y = torch.tensor(buf[1:self.seq_len+1], dtype=torch.long)
                    buf = buf[self.seq_len:]
                    shuffle_buffer.append((x, y))
                # While we have enough sequences, shuffle and emit
                while len(shuffle_buffer) >= max_shuffle_items or (force_all and len(shuffle_buffer) > 0):
                    idx = random.randrange(len(shuffle_buffer))
                    yield shuffle_buffer.pop(idx)
            for line in f:
                if f.tell() > end_offset:
                    break  # Reached end of our chunk
                line = line.strip()
                if not line:
                    continue
                buf.extend(self.tokenizer.encode(line, add_bos=False, add_eos=False))
                buf.append(self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 2)
                yield from yield_from_buffer(force_all=False)
            # Flush completely at end of data chunk
            yield from yield_from_buffer(force_all=True)"""
content = content.replace(old_code, new_code)
with open("scripts/train.py", "w") as f:
    f.write(content)
print("done")
