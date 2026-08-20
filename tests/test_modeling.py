from spider.modeling import resolve_compute_dtype


class _Cuda:
    def __init__(self, bf16: bool) -> None:
        self.bf16 = bf16

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def device_count(self) -> int:
        return 1

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (8, 9) if self.bf16 else (7, 5)


class _Torch:
    float16 = "fp16"
    bfloat16 = "bf16"

    def __init__(self, bf16: bool) -> None:
        self.cuda = _Cuda(bf16)


def test_explicit_fp16_overrides_l4_bf16_support() -> None:
    assert resolve_compute_dtype(_Torch(True), "float16") == "fp16"


def test_auto_uses_native_bf16() -> None:
    assert resolve_compute_dtype(_Torch(True), "auto") == "bf16"


def test_explicit_bf16_rejects_unsupported_device() -> None:
    try:
        resolve_compute_dtype(_Torch(False), "bfloat16")
    except RuntimeError as error:
        assert "not supported" in str(error)
    else:
        raise AssertionError("unsupported bf16 was accepted")
