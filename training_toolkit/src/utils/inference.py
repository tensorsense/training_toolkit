from training_toolkit import get_video_reader


VIDEO_READER = get_video_reader()


def process_raw_video(video_path, model, processor, gen_kwargs):
    video_clip = VIDEO_READER(video_path, 8)

    # Let's use chat template to format the prompt correctly, this time without the caption
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Provide a detailed caption for this video."},
                {"type": "video"},
            ],
        },
    ]

    # Set add_generation_prompt to add the "ASSISTANT: " at the end
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    batch = processor(
        text=prompt,
        videos=[video_clip],
        return_tensors="pt",
    ).to(model.device)

    out = model.generate(**batch, pixel_values_videos=video_clip, **gen_kwargs)
    generated_text = processor.batch_decode(out, skip_special_tokens=True)

    return generated_text
