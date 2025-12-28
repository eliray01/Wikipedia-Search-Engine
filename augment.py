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

#generate_answer takes query and retrieved_documents and uses the language model to generate an answer based on the retrieved documents
def generate_answer(query, retrieved_documents):
    #build the system prompt with the retrieved documents text
    #format the context documents with clear structure
    context_parts = []
    for i, doc in enumerate(retrieved_documents, 1):
        context_parts.append(f"Document {i}:\n{doc['text']}")
    
    context_text = "\n\n".join(context_parts)
    
    instruction_prompt = f"""You are a retrieval-augmented generation (RAG) assistant. Your task is to answer questions based solely on the provided context documents.

IMPORTANT INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the context documents below.
2. Do NOT use any external knowledge or information not present in the context.
3. Provide a clear, concise, and accurate answer based on the context.
4. If multiple documents contain relevant information, synthesize the information coherently.

CONTEXT DOCUMENTS:
{context_text}"""
    
    #create messages for the language model with system prompt and user query
    messages = [
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": query}
    ]
    
    #convert messages variable which stores the prompt message for the language model into a single prompt string which will be passed to the model to use
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
        max_new_tokens=512,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    
    #decode only the newly generated tokens (excluding the input prompt)
    #get the length of input tokens to extract only the generated part
    input_length = model_inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    #clean up the answer by removing any remaining prompt artifacts
    answer = answer.strip()
    
    #remove common prefixes that might appear
    prefixes_to_remove = ["assistant", "Assistant", "ASSISTANT"]
    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
            #remove colon if present
            if answer.startswith(":"):
                answer = answer[1:].strip()
    
    #return the generated answer
    return answer