import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os
import pandas as pd
import parser
from agent import Agent, ChatUnderstanding

load_dotenv()
token = os.getenv("DISCORD_TOKEN")

handler = logging.FileHandler(filename='discord.log',encoding='utf-8',mode='w')

intents = discord.Intents.default()

intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!',intents=intents)

agent = Agent()

@bot.event
async def on_ready():
    print("AIzter2 is running")

def to_dataframe(bot, messages : list) -> pd.DataFrame:
    # Sort messages in chronological order
    if messages[-1].created_at < messages[0].created_at:
        messages.reverse()

    content = [message.clean_content for message in messages]
    #attachments = []
    author = [message.author.name for message in messages]
    authorid = [message.author.id for message in messages]
    date = [message.created_at for message in messages]
    
    # Swap the names of the real and AI alzter
    # so that the AI alzter assumes the position
    # of the real Alzter
    for i, name in enumerate(author):
        
        if name == "alzter":
            name = "AlzterBot"
        if name == bot.user.name:
            name = "alzter"

        author[i] = name

    data = {
        "AuthorID" : authorid,
        "Author" : author,
        "Date" : date,
        "Content" : content
    }

    data = pd.DataFrame(data)

    data = parser.parse_discord_conversation(data)

    return(data)

def understand_conversation(bot, message_df : pd.DataFrame, context : str | None = None) -> ChatUnderstanding:
    """
    Pass a series of Discord messages to the LLM to
    understand them in terms of topic, relationship between users,
    and alzter's level of interest in the conversation.
    
    Args:
        message_df (DataFrame): DataFrame of messages.
        context (str): Optional additional context related to the conversation.
    Returns:
        understanding(ChatUnderstanding): Understanding of the conversation. Has three items: "topic", "relationship", "interest".
    """

    understanding = agent.understand_conversation_pov(message_df, "alzter", context=context, tokens=128)

    return understanding

def generate_reply(bot, message_history : list) -> str:
    message_df = to_dataframe(bot, message_history)

    understanding = understand_conversation(bot, message_df)
    
    response = agent.reply(message_df, context=understanding.relationship)

    return response 

@bot.event
async def on_message(message):
    
    if message.author == bot.user: return
         
    print(message.content)
    print(message.author)
    
    # await message.channel.send("hi")
    
    message_history = [message async for message in message.channel.history(limit=20)]
    response = generate_reply(bot, message_history)
    
    await message.channel.send(response)

    await bot.process_commands(message)

bot.run(token, log_handler=handler, log_level=logging.DEBUG)
