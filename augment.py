from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#specify the model name which is being used from hugging face and save it to model_name
model_name = "Qwen/Qwen2.5-3B-Instruct"

#load the lanuage model that is being used using variable model_name
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
#load the sepcific tokenizer for the model which will break the text into chunks which the model can utilise.
tokenizer = AutoTokenizer.from_pretrained(model_name)

#generated_text_to_queries takes input of the generated_text and gets the actual queries that will be used 
def generated_text_to_queries(generated_text):
    #splits each generated text into a new line and saves it to lines 
    lines = generated_text.splitlines()
    #insitalise empty list called queries which will store the final queries
    queries = []
    #intialise capture and set it to false this is to control when to start capturing liens
    capture = False
    #for loop which will loop over each line of text within lines 
    for line in lines:
        #if the current line is equal to assistant in lower case then set capture to True
        if line.strip().lower() == "assistant":
            capture = True
            continue
        #if capture is true and line.strip() (.strip() removes trailing and leading empty spaces) is not empty 
        #then append line.strip() (line with no empty spaces) to the list of queries
        if capture and line.strip():
            queries.append(line.strip())
    #return the list of queries
    return queries

#generate_augmented_queries takes query and num_augments which is 5 by default this function will use the users query and the language model to genrate similar agumented queries
def generate_augmented_queries(query, num_augments=5):
    #prompt message for the language model is created and save it to messaegs
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant whose sole task is to generate alternative queries with the same meaning as the provided query. "
                f"For any query given, output exactly {num_augments} similar queries, each on its own line, and do not include any other text or commentary."
            )
        },
        {"role": "user", "content": query}
    ]

    #convert messages variable which stores the prompt messaeg for the language model into a single prompt string which will be passed to the model to use
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    #then tokenize text variable from above and save it to model_inputs 
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    #generate text using model and pass model_inputs through the model 
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=100,
        num_return_sequences=1
    )

    #decode the generated_text tokens so it is readable and set skip_special_tokens to true to skip special symbols and save it to generated_text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #call the generated_text_to_queries and pass the generated text which will return the list of extracted queries and save it to extracted_queries_list
    extracted_queries_list = generated_text_to_queries(generated_text)
    #print extracted_queries_list
    print(extracted_queries_list)
    #return extracted_queries_list
    return extracted_queries_list