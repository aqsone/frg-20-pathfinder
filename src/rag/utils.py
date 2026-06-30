import time


def call_with_retry(fn, *args, **kwargs):
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e):
                print("  -> [!] Quota atteint. Pause de 60 secondes...")
                time.sleep(60)
            else:
                raise