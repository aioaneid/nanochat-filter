import math
from datasets import load_dataset
from tasks.common import Task

class SmolTalk(Task):
    """ smol-smoltalk dataset. train is 460K rows, test is 24K rows. """

    def __init__(self, tokenizer, split, start=0, stop=None, step=1):
        super().__init__(start=start, stop=stop, step=step)
        self.tokenizer = tokenizer
        assert split in ["train", "test"], "SmolTalk split must be train|test"

        # Load the raw dataset
        ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)

        def has_supervision(row):
            messages = row["messages"]
            conversation = {"messages": messages}
            _, mask = self.tokenizer.render_conversation(conversation)
            # mask[1:] corresponds to targets, must have at least one target '1'
            return any(m == 1 for m in mask[1:])

        # --- Precise Filtering Logic ---

        # 1. Calculate exactly how many valid rows are needed to satisfy Task slicing
        if stop is None:
            # If stop is None, Task uses the full length, so we must scan everything
            target_count = len(ds)
        else:
            # Replicate Task.__len__ logic to find the last physical index accessed
            span = stop - start
            if span <= 0:
                target_count = 0
            else:
                num_logical_examples = (span + step - 1) // step
                last_physical_index = start + (num_logical_examples - 1) * step
                # We need the array to be at least this long
                target_count = last_physical_index + 1

        # 2. Filter until we hit the target count
        filtered_indices = []
        for i in range(len(ds)):
            if has_supervision(ds[i]):
                filtered_indices.append(i)

            if len(filtered_indices) >= target_count:
                break

        self.ds = ds.select(filtered_indices)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # sanity checking asserts here
        # TODO: we could remove these asserts later, for now just don't want any footguns
        # there is an optional system message at the beginning
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:] # optional system message is OK
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, "SmolTalk messages must have at least 2 messages"
        for i, message in enumerate(rest_messages):
            # user and assistant alternate as user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"
        # ---------------------------------------------------------------------
        # create and return the Conversation object (ok to emit the system message too)
        conversation = {
            "messages": messages,
        }
        return conversation
