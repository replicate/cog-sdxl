import os

env = os.environ
env["DEBUG"] = "1"
env["PRELOAD_PATH"] = "meta.csv"
# nv_ld = "/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin:/usr/local/cuda/lib64"
# env["LD_LIBRARY_PATH"] = f"{env['LD_LIBRARY_PATH']}:{nv_ld}"
# env["PATH"]=f"{env['PATH']}:/usr/local/nvidia/bin"
env["NUM_THREADS"] = "10"
env["NUM_STREAMS"] = "40"
env["DOWNLOAD"] = "1"
env["REMOTE_CSV"] = "0"
env["DOWNLOADER_PATH"] = "/usr/local/bin/remotefile"
env["NOTIME"] = "1"
env["VMSPLICE"] = "1"
#env["SKIP_SETPIPE_SZ"] = "1"
#os.system("ln -s /boneless_model.pth /src/boneless_model.pth") # ugh
os.system("ln -s /usr/local/lib/python3.11/site-packages/torch/lib/libcudart-d0da41ae.so.11.0 /usr/lib/libcudart.so.11.0")
# SYMLINK_CUDART shouldn't be needed probably...
