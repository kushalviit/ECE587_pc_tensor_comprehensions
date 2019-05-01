import tensor_comprehensions as tc
import torch
lang = """
def tensordot(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1) -> (O) {
    O(n, c1, c3, h, w) +=! I0(n, c1, c2, h, w) * I1(n, c2, c3, h, w)
}
"""
N, C1, C2, C3, H, W = 32, 512, 8, 2, 28, 28
tensordot = tc.define(lang, name="tensordot")
I0, I1 = torch.randn(N, C1, C2, H, W).cuda(), torch.randn(N, C2, C3, H, W).cuda()
best_options = tensordot.autotune(I0, I1, cache=True)
out = tensordot(I0, I1, options=best_options)
