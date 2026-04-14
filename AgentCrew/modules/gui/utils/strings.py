from AgentCrew.modules.chat.agent_evaluation import remove_agent_evaluation


def tag_action_strip(data: str) -> str:
    if "<Tag_Action>" in data and "</Tag_Action>" in data:
        prefix = "the user request: "
        start = data.find(prefix)
        end = data.find("</Tag_Action>")
        if start != -1 and end != -1:
            return data[start + len(prefix) : end]
    return data


def agent_evaluation_remove(data: str) -> str:
    return remove_agent_evaluation(data)


def need_print_check(message: str) -> bool:
    return (
        not message.startswith("<Transfer_Tool>")
        and not message.startswith("Memories related to the user request:")
        and not message.startswith("Need to tailor response bases on this")
    )
