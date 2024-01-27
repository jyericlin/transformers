from transformers import AutoTokenizer, OPTForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import json
from loguru import logger
import numpy as np

def get_alpaca_data(json_file_path):
    '''
    Example:
    [{
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    },...]
    '''
    PROMPT_TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: "
    prompt_list = []
    json_data = json.load(open(json_file_path, "r"))
    for data in json_data:
        prompt = PROMPT_TEMPLATE + data["instruction"]
        if data["input"] != "":
            prompt += " ### Input: " + data["input"]
        prompt_list.append(prompt)
    return prompt_list
    
def setup_opt_model(model_name, use_cuda):
    model = OPTForCausalLM.from_pretrained(model_name, torchscript=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    if use_cuda:
        model.cuda()
    
    return model, tokenizer

def setup_t5_model(model_name, use_cuda):
    model = T5ForConditionalGeneration.from_pretrained(model_name, torchscript=True)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model.eval()
    if use_cuda:
        model.cuda()
    
    return model, tokenizer



def run(model_name=None, prompt_data_path=None, use_cuda=False, repeat_time=10):
    if model_name is None:
        # model_name = "facebook/opt-350m"
        model_name = "facebook/opt-2.7B"
        # model_name = "t5-small"

    # model 
    model, tokenizer = setup_opt_model(model_name, use_cuda)
    # model, tokenizer = setup_t5_model(model_name, use_cuda)

    # inputs = tokenizer("Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:")
    # print(inputs.input_ids)
    # exit()

    # prompts
    if prompt_data_path is None:
        prompt_data_path = "./alpaca_data.json"
    prompts = get_alpaca_data(prompt_data_path)

    # start measurement
    measurement_results = []
    for prompt in prompts:
        logger.info("Processing prompt: " + prompt)
        measurement_data = {}

        inputs = tokenizer(prompt, return_tensors="pt")
        if use_cuda:
            inputs = inputs.to("cuda")

        input_length = len(inputs.input_ids[0])
        measurement_data["input_length"] = input_length
        measurement_data["prompt"] = prompt

        logger.info(f"input_length: {input_length}")
    
        # run model once to get the cache
        max_length = input_length + 1
        # warm up
        t_no_use_cache_1 = time.time()
        generate_ids, _ = model.generate(inputs.input_ids, max_length=max_length, use_cache=False)
        t_no_use_cache_2 = time.time()
        logger.debug(f"no use cache generate time: {t_no_use_cache_2-t_no_use_cache_1}")

        generate_ids, model_kwargs = model.generate(inputs.input_ids, max_length=max_length, use_cache=True)

        exec_times = []
        for i in range(repeat_time):
            t_no_cache_1 = time.time()
            generate_ids, _ = model.generate(inputs.input_ids, max_length=max_length, use_cache=True)
            t_no_cache_2 = time.time()
            exec_times.append(t_no_cache_2-t_no_cache_1)
        
        no_cache_generate_time_avg = np.average(exec_times)
        no_cache_generate_time_std = np.std(exec_times)

        no_cache_measure_data = measurement_data.copy()
        no_cache_measure_data["generate_time_avg"] = no_cache_generate_time_avg
        no_cache_measure_data["generate_time_std"] = no_cache_generate_time_std
        no_cache_measure_data["cache_seq_length"] = 0
        no_cache_measure_data["input_ids"] = inputs.input_ids
        logger.debug(f"no cache generate time: {no_cache_generate_time_avg}, std: {no_cache_generate_time_std}")
        measurement_results.append(no_cache_measure_data)
        print(no_cache_measure_data)


        # generate with cache

        # curr_cache_seq_length = input_length - 1
        # min_length = 1
        curr_cache_seq_length = 21
        min_length = 20

        while curr_cache_seq_length >= min_length:
            seq_measurement_data = measurement_data.copy()
            seq_measurement_data["cache_seq_length"] = curr_cache_seq_length
            seq_measurement_data["input_ids"] = inputs.input_ids[:, :curr_cache_seq_length]

            new_past_key_values = []

            for past_key_value in model_kwargs["past_key_values"]:
                new_past_key_values.append((past_key_value[0][:, :, :curr_cache_seq_length, :].clone(), past_key_value[1][:, :, :curr_cache_seq_length, :].clone()))

            new_past_key_values = tuple(new_past_key_values)
            logger.debug("new_past_key_values" + str(new_past_key_values[0][0].shape))

            exec_times = []
            for i in range(repeat_time):
                t_rg_1 = time.time()
                generate_ids, _ = model.generate(inputs.input_ids, max_length=max_length, use_cache=True, past_key_values=new_past_key_values)
                t_rg_2 = time.time()
                exec_times.append(t_rg_2-t_rg_1)

            generation_time_avg = np.average(exec_times)
            generation_time_std = np.std(exec_times)


            logger.debug(f"generate time avg: {generation_time_avg}, std: {generation_time_std}")
            result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            logger.debug(result)

            curr_cache_seq_length = curr_cache_seq_length - 1

            seq_measurement_data["generate_time_avg"] = generation_time_avg
            seq_measurement_data["generate_time_std"] = generation_time_std
            seq_measurement_data["generate_times"] = exec_times

            measurement_results.append(seq_measurement_data)
            print(seq_measurement_data)
        break



if __name__ == "__main__":
    run(repeat_time=3)