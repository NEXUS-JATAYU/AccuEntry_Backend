from agents import data_capture_agent

async def handle_chat( session_id : str , user_input : str):
    return await data_capture_agent.run(session_id,user_input)