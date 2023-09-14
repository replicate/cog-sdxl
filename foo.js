function work(prompt, v) {
    return v + 1;
}

function pipeline(prompt, callback, callback_steps, number_inference_steps) {
    let v = 0;
    for (var i = 0; i < number_inference_steps; i++) {
        v = work(prompt, v);
        if (i % callback_steps == 0)
            callback(v)
    }
    return v;
}

