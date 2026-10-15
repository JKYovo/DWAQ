"""Process-local deadlines for quality-v3's native lifecycle/config uploads.

Scalar uploads use the separately supervised TensorBoard repair process.
The installed SDK adapter uses setdefault(timeout), which does not replace
requests' explicit timeout=None. No site-package files are changed here.
"""


def bound_native_requests():
    from swanlab.sdk.internal.pkg.client.session import SessionWithRetry
    original = SessionWithRetry.request
    if getattr(original, "_elf3_bounded", False):
        return

    def bounded(self, method, url, *args, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (5, 25)
        if kwargs.get("retries") is None:
            kwargs["retries"] = 0
        return original(self, method, url, *args, **kwargs)

    bounded._elf3_bounded = True
    SessionWithRetry.request = bounded
