#import asyncio
#import os
#import random
#from livekit import api
#from dotenv import load_dotenv

#load_dotenv()

#async def main():
    #lkapi = api.LiveKitAPI()
    
    # Unique room name for this specific call
    #room_name = f"outbound-{random.randint(1000, 9999)}"
    
    # YOUR DETAILS
    #my_number = "+447XXXXXXXXX" # REPLACE WITH YOUR O2 NUMBER
    #trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID") 

    #print(f"🚀 Xhemal is about to call you at {my_number}...")

    # 1. Dispatch the Agent to the room
    #await lkapi.agent_dispatch.create_dispatch(
        #api.CreateAgentDispatchRequest(
        #    agent_name="xhemal",
    #        room=room_name
    #    )
    #)

    # 2. Dial your phone and bridge it to the room
    #await lkapi.sip.create_sip_participant(
     #   api.CreateSIPParticipantRequest(
          #  sip_call_to=my_number,
         #   room_name=room_name,
          #  participant_identity="O2_Mobile_User",
      ###      sip_trunk_id=trunk_id,
           # wait_until_answered=True # Don't start the agent until you pick up
       # )
   # )

    #await lkapi.aclose()
    #print("📞 Outbound request sent. Pickup your phone!")

#if __name__ == "__main__":
 #   asyncio.run(main())