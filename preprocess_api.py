import replicate

output = replicate.run(
    "replicate/sdxl_preprocess:bd1158a5052ed46176da900ad7e2a80ea04a3c46196d93f9e1db879fd1ce7f29",
    input={
        "files": open("./zeke.zip", "rb"),
        "caption_text": "a photo of TOK",
        "mask_target_prompts": "a face of a man",
        "target_size": 1024,
    },
)

print(output)
