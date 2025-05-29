from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import warnings

def parse_discord_conversation(c : pd.DataFrame) -> pd.DataFrame:
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

def split_by_conversations(messages : pd.DataFrame, gap_mins : int = 50, min_conv_length : int = 5) -> list[pd.DataFrame]:
    """
    Split a Discord conversation history into a list of shorter conversations separated by gap_mins minutes.

    Args:
        messages (DataFrame): Discord conversation history.
        gap_mins (int): How many minutes must have elapsed since the last message for the current message to be treated as the start of a new conversation.
        min_conv_length (int): If a conversation has less messages than this, don't include it in the list.
        max_conv_length (int): If a conversation has more messages than this, slice it up into chunks of this size.

    Returns:
        conversations (list[DataFrame]): List of all conversations ordered from least to most recent.
    """
    def is_new_conversation(delay : timedelta, max_delay_mins : int = gap_mins):
        """
        Given a delay between messages, asses whether the delay is sufficient enough
        for the message to be considered the start of a new conversation.
        """
        max_delay = timedelta(minutes = max_delay_mins)
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
        if not target_user in users:
            users = np.insert(users, 0, target_user)

    string = "Conversation between " + ", ".join(users) + "."

    string += "\nObtained " + end_time.strftime("%y/%m/%d, %H:%M:%S") + "."

    return string

def to_string(messages : pd.DataFrame, header : bool = False, timestamp : bool = True, anonymise : bool = False, target_user : str | None = None, context : str | None = None) -> str:
    """
    Convert a Discord conversation history from DataFrame into a raw string.

    Args:
        messages (DataFrame): The conversation.
        header (bool, optional): Whether to include metadata at the start of the string, including the users in conversation and the time of the last message. Defaults to True.
        timestamp (bool, optional): Whether to include a timestamp for each message. Defaults to True.
        anonymise (bool, optional): Whether to replace all user names with "user", except for the target user. Defaults to False.
        target_user (str, optional): Which user to use as the focal point of the conversation. Defaults to None.
        context (str, optional): Additional context for the conversation. If provided, added after the header. Defaults to None.
    """
    string = ""

    if header:
        string += get_header(messages, target_user=target_user)

    if type(target_user) is str: target_user = target_user.lower()

    if context: string += "\nContext of the conversation:\n" + context

    for i, message in messages.iterrows():
        author = message.Author if (not anonymise or message.Author.lower() == target_user) else "user"
        string += "\n\n" + author
        if timestamp: string += " " + message.Date.strftime("%H:%M:%S")
        string += f"\n{message.Content}"

    return string.strip()

def to_dataset(messages : pd.DataFrame | list[pd.DataFrame], target_user : str, context_length : int = 10, anonymise : bool = True, timestamp : bool = False, ignore_repeats : bool = False) -> pd.DataFrame:
    """
    Convert a Discord conversation or list of conversations into a supervised Q&A dataset from the perspective of a given user.
    This dataset can then be used to fine-tune a PLM to impersonate the user.

    A sliding window method is used to slice the data so that each input has no more than ``context_length messages``.

    Args:
        messages (DataFrame | list[DataFrame]): Discord message history / conversations to parse.
        target_user (str): Which user you want to predict the messages of.
        context_length (int, optional): Maximum number of input messages per data sample. Defaults to 10.
        anonymise (bool, optional): Replaces the usernames of all other users with "user". Defaults to True.
        timestamp (bool, optional): Adds a timestamp at the start of each message. Defaults to False.
        ignore_repeats (bool, optional): Ignores all messages from the target user sent after a message from the same user.
        
    Returns:
        dataset (DataFrame): A Q&A dataset made out of the Discord conversations. Has two columns, "content" (inputs) and "labels" (output message).
    """
    data = {"content" : [], "label" : []}
    
    # If we're dealing with a list of messages,
    # call this function recursively.
    if type(messages) is list:

        data = pd.DataFrame(data)
        for message in messages:
            
            # Process the messages sequentially
            message_data = to_dataset(message, target_user, context_length, anonymise, timestamp, ignore_repeats)

            # Append each message data to the end of the dict
            data = pd.concat([data, message_data])
        
        data = data.reset_index(drop=True)
        return data

    elif type(messages) is pd.DataFrame:

        # Get all messages from the target user
        target_user_indices = list(messages[messages.Author == target_user].index)
        other_user_indices = list(messages[messages.Author != target_user].index)
        
        # If there's no messages by the user and others, return nothing
        if not target_user_indices or not other_user_indices:
            return pd.DataFrame(data)

        # Remove all messages from target user which aren't preceded by
        # a different user's message at the start of the conversation.
        target_user_indices = [i for i in target_user_indices if i > other_user_indices[0]]

        # Remove all messages from target user which
        # aren't preceded by a different user.
        if ignore_repeats:
            target_user_indices = [index for i, index in enumerate(target_user_indices) if index - target_user_indices[i-1] > 1 or i == 0]

        # Use a sliding window to slice the conversation up.
        for msgs in messages.rolling(context_length):
            # Ignore all slices which don't end with a message by the target user
            if msgs.index.stop - 1 not in target_user_indices: continue
        
            inputs = msgs.iloc[:-1]
            output = msgs.iloc[-1]

            # Convert inputs/outputs to string
            inputs = to_string(inputs, header=False, timestamp=timestamp, anonymise=anonymise, target_user=target_user)
            output = output.Content

            data['content'].append(inputs)
            data['label'].append(output)

        return pd.DataFrame(data)

    raise ValueError("Messages must be DataFrame or List")

def create_dataset(files : list[str], target_user : str, anonymise : bool = True, input_length : int = 30, packing : bool = True, timestamp : bool = False, conversation_split_mins : int | None = 120) -> pd.DataFrame:
    """
    Convert a list of Discord conversations into a supervised Q&A dataset from the perspective of a given user.
    This dataset can then be used to fine-tune a PLM to impersonate the user.

    Args:
        files (list[str]): List of Discord conversation history CSV files exported with DiscordChatExporter.
        target_user (str): Which user to predict the messages for.
        anonymise (bool, optional): Replaces usernames of all other users with "user".
                                    Enable if you want the LLM to act uniformly for all users. Defaults to True.
        input_length (int, optional): For each sample, how many messages to include before the target user's message for context.
                                      Greater values allow the LLM to infer more context from each conversation. Defaults to 30.
        packing (bool, optional): Concatenates all consecutive messages by the target user using line breaks to reduce dataset size.
                                  If enabled, dataset will be smaller but LLM responses will be larger. Defaults to True.
        timestamp (bool, optional): Includes a timestamp within each input message in the dataset. Defaults to False.
        conversation_split_mins (int | None, optional): If given, slices each message history into separated conversations where the time between
                                                        each conversation in minutes is greater than the given amount. Defaults to 120.
    
    Returns:
        dataset (DataFrame): A Q&A dataset made out of the Discord conversations. Has two columns, "content" (inputs) and "labels" (output message).
    """
    data = pd.DataFrame({"content": [], "label": []})
    
    for file in tqdm(files, "Preprocessing Discord Conversations"):

        try:
            chat = parse_discord_conversation_csv(file)
            if conversation_split_mins: chat = split_by_conversations(chat, conversation_split_mins)
            if packing: chat = group_consecutive(chat, target_user)
            chat = to_dataset(chat, target_user, input_length, ignore_repeats=packing, anonymise=anonymise, timestamp=timestamp)
            data = pd.concat([data, chat])

        except Exception as e:
            warnings.warn(f"Error reading Discord conversation history file at path {file}. Traceback: {str(e)}. Ignoring the file and proceeding.")
        
    data = data.reset_index(drop=True)

    return data
