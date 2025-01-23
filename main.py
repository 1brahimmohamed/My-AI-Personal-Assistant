import os
import dotenv
import telebot
import torch
from transformers import pipeline

dotenv.load_dotenv()
BOT_TOKEN = os.environ.get('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device=0 if torch.cuda.is_available() else -1 ## Use GPU if available
)

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    """
    Send welcome message to user when they start the bot
    """
    bot.reply_to(message, "Howdy, I am your personal assistant. how can I help you today?")


@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    """
    Respond to user message using the TinyLlama about animals
    """
    prompt = pipe.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a friendly chatbot who always responds in the style of an expert"},
            {"role": "user", "content": message.text},
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    generated_text = outputs[0]["generated_text"]
    assistant_response = generated_text.split("<|assistant|>")[-1].strip()

    bot.reply_to(message, assistant_response)

# Run the bot
bot.infinity_polling()