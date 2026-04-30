from pathlib import Path

from .packed_data import start_count, token_count


class MappedPackedDataset:
    def __init__(self, cache_dir: Path, sequence_len: int, pad_id: int):
        import torch
        from torch.utils.data import Dataset

        tokens_path = cache_dir / "tokens.bin"
        mask_path = cache_dir / "loss_mask.bin"
        starts_path = cache_dir / "starts.bin"
        token_count_value = token_count(tokens_path)
        mask_count_value = mask_path.stat().st_size if mask_path.exists() else 0
        start_count_value = start_count(starts_path)
        tokens = torch.from_file(str(tokens_path), shared=False, size=token_count_value, dtype=torch.uint16)
        mask = torch.from_file(str(mask_path), shared=False, size=mask_count_value, dtype=torch.uint8)
        starts = torch.from_file(str(starts_path), shared=False, size=start_count_value, dtype=torch.int64)

        class _Dataset(Dataset):
            def __len__(self_inner):
                return start_count_value

            def __getitem__(self_inner, index):
                start = int(starts[index].item())
                end = start + sequence_len + 1
                ids = tokens[start:end].to(dtype=torch.long)
                loss_mask = mask[start:end].to(dtype=torch.bool)
                if ids.numel() < sequence_len + 1:
                    missing = sequence_len + 1 - ids.numel()
                    ids = torch.cat([ids, torch.full((missing,), pad_id, dtype=torch.long)])
                    loss_mask = torch.cat([loss_mask, torch.zeros(missing, dtype=torch.bool)])
                labels = ids[1:].clone()
                labels.masked_fill_(~loss_mask[1:], -100)
                return ids[:-1].contiguous(), labels.contiguous()

        self._dataset = _Dataset()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]


class BatchMappedPackedDataset:
    def __init__(self, cache_dir: Path, sequence_len: int, pad_id: int, batch_size: int):
        import torch
        from torch.utils.data import Dataset

        tokens_path = cache_dir / "tokens.bin"
        mask_path = cache_dir / "loss_mask.bin"
        starts_path = cache_dir / "starts.bin"
        token_count_value = token_count(tokens_path)
        mask_count_value = mask_path.stat().st_size if mask_path.exists() else 0
        start_count_value = start_count(starts_path)
        tokens = torch.from_file(str(tokens_path), shared=False, size=token_count_value, dtype=torch.uint16)
        mask = torch.from_file(str(mask_path), shared=False, size=mask_count_value, dtype=torch.uint8)
        starts = torch.from_file(str(starts_path), shared=False, size=start_count_value, dtype=torch.int64)
        batches = max(1, start_count_value // max(1, batch_size))

        def window(start: int):
            end = start + sequence_len + 1
            ids = tokens[start:end].to(dtype=torch.long)
            loss_mask = mask[start:end].to(dtype=torch.bool)
            if ids.numel() < sequence_len + 1:
                missing = sequence_len + 1 - ids.numel()
                ids = torch.cat([ids, torch.full((missing,), pad_id, dtype=torch.long)])
                loss_mask = torch.cat([loss_mask, torch.zeros(missing, dtype=torch.bool)])
            labels = ids[1:].clone()
            labels.masked_fill_(~loss_mask[1:], -100)
            return ids[:-1].contiguous(), labels.contiguous()

        class _Dataset(Dataset):
            def __len__(self_inner):
                return batches

            def __getitem__(self_inner, index):
                base = int(index) * batch_size
                rows = [window(int(start.item())) for start in starts[base : base + batch_size]]
                return torch.stack([row[0] for row in rows]), torch.stack([row[1] for row in rows])

        self._dataset = _Dataset()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]


class SyntheticPackedDataset:
    def __init__(self, windows: int, sequence_len: int, vocab_size: int, seed: int):
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __len__(self_inner):
                return windows

            def __getitem__(self_inner, index):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed + int(index))
                ids = torch.randint(5, vocab_size, (sequence_len + 1,), dtype=torch.long, generator=generator)
                labels = ids[1:].clone()
                return ids[:-1].contiguous(), labels.contiguous()

        self._dataset = _Dataset()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]
