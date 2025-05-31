from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import re
import glob
from tqdm import tqdm
import warnings

def clean_message_content(content : str) -> str:
    """
    Remove all user mentions, hyperlinks, and whitespace from Discord message text.
    Credit: [FlintSH](https://github.com/FlintSH/Disclone/blob/3e026ca28909913fbc8e9ca44c21b8d89c5b761b/main.py#L96C1-L106C27)
    """
    if type(content) is not str: return content

    # Remove mentions (e.g., <@123456789>)
    content = re.sub(r'<@!?\d+>', '', content)
    
    # Remove links
    content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
    
    # Remove extra whitespace
    content = ' '.join(content.split())
    
    return content.strip()

def parse_discord_conversation(c : pd.DataFrame) -> pd.DataFrame:
    c["Content"] = c["Content"].map(clean_message_content)
    c["Content"] = c["Content"].replace("", np.nan)

    def parse_time_str(time : str) -> datetime:
        return datetime.strptime(time,'%Y-%m-%dT%H:%M:%S.%f0%z')

    if type(c["Date"][0]) is str:
        c["Date"] = c["Date"].map(parse_time_str)
    c["Delay"] = c["Date"] - c["Date"].shift(1)
    c["Delay"] = c["Delay"].fillna( timedelta(seconds=0) )
    c = c.dropna(subset="Content").reset_index(drop=True)
    return c

def parse_discord_conversation_csv(path : str) -> pd.DataFrame:
    conversation = pd.read_csv(path)
    return parse_discord_conversation(conversation)

def get_chat(user : str, path : str) -> pd.DataFrame:
    path = path + "/Direct Messages - " + user + " [*.csv"
    conversation = glob.glob(path)

    if conversation:
        return parse_discord_conversation_csv(conversation[0])
    else:
        raise FileNotFoundError(f"No conversation(s) found at path {path}")

def group_consecutive(messages : pd.DataFrame | list[pd.DataFrame], users : str | list[str] | None = None, delimiter : str = '\n') -> pd.DataFrame | list[pd.DataFrame]:
    """
    Given a Discord conversation history, combine all
    consecutive messages by the same user into single
    messages delimited with line breaks by default.

    Args:
        messages (DataFrame | list[DataFrame]): The conversation(s) to combine consecutive messages for.
        users (str | list[str] | None, optional): Which users to combine messages for. Defaults to all users.
        delimiter (str, optional): What string to use when concatenating messages. Defaults to "\n".
        
    Returns:
        grouped_msgs (DataFrame): The conversation history with consecutive messages combined.
    """
    # If we're dealing with a list of messages,
    # call this function recursively.
    if type(messages) is list:
        msgs = []
        for i in messages:
            concat = group_consecutive(i,users,delimiter)
            if len(concat) > 0: msgs.append(concat)
        return msgs
    
    elif type(messages) is pd.DataFrame:

        if not users: users = messages.Author.unique()
        if type(users) is str: users = [users]
        
        grouped_msgs = []
        chunk = None

        for i, message in messages.iterrows():
            prev_msg = messages.iloc[i - 1] if i > 0 else None
            prev_author = prev_msg.Author if prev_msg is not None else None

            if chunk is not None:
                if message.Author == prev_author:
                    chunk.Content += delimiter + message.Content
                else:
                    grouped_msgs.append(chunk.to_dict())
                    chunk = None

            if chunk is None:
                if message.Author in users:
                    chunk = message
                else:
                    grouped_msgs.append(message.to_dict())
                
            # if message.Author == user:
            #     if prev_msg.Author == user:
            #         chunk.Content += delimiter + message.Content
            #     else:
            #         chunk = message
            # else:
            #     if chunk is not None:
            #         grouped_msgs.append(chunk.to_dict())
            #         chunk=None
            #     grouped_msgs.append(message.to_dict())

        return pd.DataFrame(grouped_msgs)

    raise ValueError("Messages must be DataFrame or List")

def split_by_conversations(messages : pd.DataFrame, max_context_time : timedelta = timedelta(minutes=50), min_conv_length : int = 5) -> list[pd.DataFrame]:
    """
    Split a Discord conversation history into a list of shorter conversations separated by a given maximum context time.

    Args:
        messages (DataFrame): Discord conversation history.
        max_context_time (timedelta, optional): Maximum delay between messages to be treated as one conversation. Defaults to timedelta(minutes=50).
        min_conv_length (int, optional): If a conversation has less messages than this, don't include it in the list. Defaults to 5.

    Returns:
        conversations (list[DataFrame]): List of all conversations ordered from least to most recent.
    """
    def is_new_conversation(delay : timedelta, max_delay : timedelta = max_context_time):
        """
        Given a delay between messages, assess whether the delay is sufficient enough
        for the message to be considered the start of a new conversation.
        """
        return delay > max_delay

    def get_conversation_indices(messages : pd.DataFrame) -> list[list[int]]:
        """
        Given a Discord conversation history with a boolean column "Start"
        denoting the start of a conversation, return a 2D list containing
        the indices of all messages grouped by conversation.

        Args:
            messages (DataFrame): Discord conversation history with "Start" column.
        Returns:
            conversation_indices (list[list[int]])
        """
        start_indices = messages[messages["Start"] == True].index

        indices = []
        for i in range(len(start_indices)):
            if i >= len(start_indices) - 1: continue
            indices.append(
                list(range(start_indices[i], start_indices[i+1]))
            )
        return indices

    messages["Start"] = messages["Delay"].map(is_new_conversation)
    messages.loc[0, "Start"] = True

    conversation_indices = get_conversation_indices(messages)

    conversations = []
    for indices in conversation_indices:

        if len(indices) < min_conv_length: continue
        conversation = messages.iloc[indices].reset_index(drop=True)
        conversations.append(conversation)

    return conversations

def get_header(messages : pd.DataFrame, target_user : str | None = None) -> str:
    """
    Return statistics about a Discord conversation (i.e., users involved, time)
    """
    end_time = messages.iloc[-1].Date

    users = messages.Author.unique()
    if target_user:
        if target_user not in users:
            users = np.insert(users, 0, target_user)

    string = "Conversation between " + ", ".join(users) + "."

    string += "\nObtained " + end_time.strftime("%y/%m/%d, %H:%M:%S") + "."

    return string

def to_string(messages : pd.DataFrame, header : bool = False, timestamps : bool = True, usernames : bool = True, target_user : str | None = None, context : str | None = None) -> str:
    """
    Convert a Discord conversation history from DataFrame into a raw string.

    Args:
        messages (DataFrame): The conversation.
        header (bool, optional): Whether to include metadata at the start of the string, including the users in conversation and the time of the last message. Defaults to True.
        timestamps (bool, optional): Whether to include a timestamp for each message. Defaults to True.
        usernames (bool, optional): Include a username for each message. Defaults to True.
        target_user (str, optional): Which user to use as the focal point of the conversation. Defaults to None.
        context (str, optional): Additional context for the conversation. If provided, added after the header. Defaults to None.
    """
    string = ""

    if header:
        string += get_header(messages, target_user=target_user)

    if type(target_user) is str: target_user = target_user.lower()

    if context: string += "\nContext of the conversation:\n" + context

    for i, message in messages.iterrows():
        author = message.Author if (usernames or message.Author.lower() == target_user) else "user"
        string += "\n\n" + author
        if timestamps: string += " " + message.Date.strftime("%H:%M:%S")
        string += "\n" + message.Content

    return string.strip()

def to_chat_template(messages : pd.DataFrame, target_user : str, system_prompt : str = "", usernames : bool = False, timestamps : bool = False) -> dict:
    """
    Convert a Discord conversation history from a DataFrame into an OpenAI chat template.

    Args:
        messages (DataFrame): The conversation.
        target_user (str, optional): Which user to replicate the messages of. 
        system_prompt (str, optional): Recommended. Instructions for an LLM to replicate the target user's behaviour provided at the start of each conversation. Defaults to "". 
        usernames (bool, optional): Include a username for each message. Defaults to False.
        timestamps (bool, optional): Include a timestamp for each message. Defaults to True.

    Returns:
        chat_template (dict): The conversation converted to an OpenAI chat template.
    """

    chat_messages = []
    system_prompts = [system_prompt]

    metadata = []
    if usernames: metadata.append("username")
    if timestamps: metadata.append("timestamp")
    metadata = " ".join(metadata)
    if metadata:
        system_prompts.append(f"Messages are formatted as <{metadata}>:\n<message>")
    
    system_prompt = "\n\n".join(system_prompts).strip()
    
    if system_prompt:
        chat_messages.append({"role":"system","content":system_prompt})
    
    for i, message in messages.iterrows():
        role = "assistant" if message.Author.lower() == target_user.lower() else "user"
        
        # Add message user and timestamps if needed
        content = ""
        metadata = []
        if usernames: metadata.append(message.Author)
        if timestamps: metadata.append(message.Date.strftime("%H:%M:%S"))
        
        if metadata:
            metadata = " ".join(metadata) + ":"
            content += metadata + "\n"
            
        content += message.Content

        message = {"role":role,"content":content}
        chat_messages.append(message)

    return {"messages" : chat_messages}

def to_dataset(messages : pd.DataFrame | list[pd.DataFrame], target_user : str, system_prompt : str = "", context_length : int = 50, usernames : bool = False, timestamps : bool = False, max_context_time : timedelta | None = timedelta(minutes=120), packing : bool = True) -> list[dict]:
    """
    Convert a Discord conversation or list of conversations into a supervised Q&A dataset from the perspective of a given user.
    This dataset can then be used to fine-tune a PLM to impersonate the user.
    
    A sliding window method is used to slice the data so that each input has no more than ``context_length messages``.
    
    Args:
        messages (DataFrame | list[DataFrame]): Discord message history / conversations to parse.
        target_user (str): Which user you want to predict the messages of.
        system_prompt (str, optional): Recommended. Instructions for an LLM to replicate the target user's behaviour provided at the start of each conversation. Defaults to "". 
        context_length (int, optional): Maximum number of input messages per data sample. Defaults to 50.
        usernames (bool, optional): Include a username for each message. Defaults to False.
        timestamps (bool, optional): Include a timestamp for each message. Defaults to False.
        max_context_time (timedelta | None, optional): If given, slices each message history into separated conversations where the time between
                                                       each conversation is greater than the given amount. Defaults to timedelta(minutes=120).
        packing (bool, optional): Concatenates all consecutive messages by the target user using line breaks to reduce dataset size.
                                  If enabled, dataset will be smaller but LLM responses will be larger. Defaults to True.
        
    Returns:
        dataset (list[dict]): A Q&A dataset made out of the Discord conversations in OpenAI chat format.
    """
    
    if type(messages) is pd.DataFrame:
        if max_context_time is not None:
            if "Start" not in messages:
                messages = split_by_conversations(messages, max_context_time)
        if packing: messages = group_consecutive(messages, target_user)
    # 
    data = []
    
    # If we're dealing with a list of messages,
    # call this function recursively.
    if type(messages) is list:

        for message in messages:
            # Process the messages sequentially
            message_data = to_dataset(message, target_user=target_user, system_prompt=system_prompt, context_length=context_length, usernames=usernames, timestamps=timestamps, max_context_time=max_context_time) 
            
            # Append each message data to the end of the dict
            data.extend(message_data) 
         
        return data

    elif type(messages) is pd.DataFrame:

        # Get all messages from the target user
        target_user_indices = list(messages[messages.Author == target_user].index)
        other_user_indices = list(messages[messages.Author != target_user].index)
        
        # If there's no messages by the user and others, return nothing
        if not target_user_indices or not other_user_indices:
            return [] 

        # Remove all messages from target user which aren't preceded by
        # a different user's message at the start of the conversation.
        target_user_indices = [i for i in target_user_indices if i > other_user_indices[0]]

        # Remove all messages from target user which
        # aren't preceded by a different user.
        #if packing:
        #    target_user_indices = [index for i, index in enumerate(target_user_indices) if index - target_user_indices[i-1] > 1 or i == 0]

        # Use a sliding window to slice the conversation up.
        for msgs in messages.rolling(context_length):
            # Ignore all slices which don't end with a message by the target user
            if msgs.index.stop - 1 not in target_user_indices: continue
             
            # Convert inputs/outputs to string
            chat = to_chat_template(msgs, system_prompt=system_prompt, target_user=target_user, usernames=usernames, timestamps=timestamps)
            
            data.append(chat)
        
        return data

    raise ValueError("Messages must be DataFrame or List")

def create_dataset(files : str | list[str], target_user : str, system_prompt : str = "", usernames : bool = False, timestamps : bool = False, context_length : int = 50, max_context_time : timedelta | None = timedelta(minutes=120), packing : bool = True) -> list[dict]:
    """
    Convert a list of Discord conversations into a supervised Q&A dataset from the perspective of a given user.
    This dataset can then be used to fine-tune a PLM to impersonate the user.

    Args:
        files (list[str]): List of Discord conversation history CSV files exported with DiscordChatExporter.
        target_user (str): Which user to predict the messages for.
        system_prompt (str, optional): Recommended. Instructions for an LLM to replicate the target user's behaviour provided at the start of each conversation. Defaults to "". 
        usernames (bool, optional): Includes the username of each user in their message. 
                                    Disable if you want the LLM to respond uniformly for all users. Defaults to True.
        timestamps (bool, optional): Includes a timestamp within each input message in the dataset. Defaults to False.
        context_length (int, optional): For each sample, how many messages to include before the target user's message for context.
                                      Greater values allow the LLM to infer more context from each conversation. Defaults to 50.
        max_context_time (timedelta | None, optional): If given, slices each message history into separated conversations where the time between
                                                       each conversation is greater than the given amount. Defaults to timedelta(minutes=120).
        packing (bool, optional): Concatenates all consecutive messages by the target user using line breaks to reduce dataset size.
                                  If enabled, dataset will be smaller but LLM responses will be larger. Defaults to True.
    
    Returns:
        dataset (DataFrame): A Q&A dataset made out of the Discord conversations. Has two columns, "content" (inputs) and "labels" (output message).
    """

    if type(files) is str: files = [files]

    data = [] 
    
    for file in tqdm(files, "Preprocessing Discord Conversations"):

        try:
            chat = parse_discord_conversation_csv(file)
            chat = to_dataset(chat, target_user, system_prompt=system_prompt, context_length=context_length, max_context_time=max_context_time, packing=packing, usernames=usernames, timestamps=timestamps)
            data.extend(chat)

        except Exception as e:
            warnings.warn(f"Error reading Discord conversation history file at path {file}. Traceback: {str(e)}. Ignoring the file and proceeding.")
        
    return data
