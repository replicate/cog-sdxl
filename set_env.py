import os

env = {
    # this is just to recompress
    # "LOAD_UNCOMPRESSED" : "1",
    # "NO_PRELOAD" : "1",
    "NVCOMP_LOG_LEVEL": "2",  # warn
    "NVCOMP_LOG_FILE": "stderr",
    "DEBUG" : "1",
    "PRELOAD_PATH": "meta.csv",
    # nv_ld = "/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin:/usr/local/cuda/lib64"
    # "LD_LIBRARY_PATH" : f"{env['LD_LIBRARY_PATH']}:{nv_ld}",
    # "PATH"=:"{env['PATH']}:/usr/local/nvidia/bin",
    "NUM_THREADS": "20",
    "NUM_STREAMS": "40",
    "DOWNLOAD": "1",
    "REMOTE_CSV": "0",
    # "DOWNLOADER_PATH": "/usr/local/bin/remotefile",
    # "DOWNLOADER_PATH": "/src/remotefile",
    # "DOWNLOADER_PATH" : "/usr/bin/curl",
    "DOWNLOADER_PATH" : "/src/download.sh",
    "NOTIME": "1",
    # "VMSPLICE" : "1",
    # "SKIP_SETPIPE_SZ" : "1",
    **os.environ # allow setting envvars to override these settings
}
os.environ.update(env)
# os.system("ln -s /boneless_model.pth /src/boneless_model.pth") # ugh
os.system("ln -s /usr/local/lib/python3.11/site-packages/torch/lib/libcudart-d0da41ae.so.11.0 /usr/lib/libcudart.so.11.0")
# SYMLINK_CUDART shouldn't be needed probably...
os.system(
    "ln -s /lib/python3.11/site-packages/torch/lib/libcudart* /lib/python3.11/site-packages/nyacomp.libs/libcudart.so.11.0"
)
