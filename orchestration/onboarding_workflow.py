from data_capture import data_capture
# from core.agent import AgentState

async def handle_chat( session_id : str , user_input : str):
    return await data_capture.run(session_id, user_input)