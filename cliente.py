import asyncio
import uuid

import httpx

from a2a.client import A2ACardResolver, create_client
from a2a.helpers import get_artifact_text, get_message_text
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)


SERVER_URL = "http://127.0.0.1:9999"


async def main():

    async with httpx.AsyncClient() as http_client:

        # Descubrir AgentCard
        resolver = A2ACardResolver(
            http_client,
            SERVER_URL,
        )

        card = await resolver.get_agent_card()

        print("Agente:", card.name)
        print("Capacidades disponibles:")

        for skill in card.skills:
            print(f"- {skill.id}: {skill.name}")

        # Crear el cliente A2A
        client = await create_client(card)

        # Mantener el mismo contexto durante la conversación
        context_id = str(uuid.uuid4())

        print("\nEscribe 'salir' para terminar.")

        try:
            while True:
                pregunta = input("\nTú: ").strip()

                if pregunta.lower() == "salir":
                    break

                if not pregunta:
                    continue

                mensaje = Message(
                    role=Role.ROLE_USER,
                    message_id=str(uuid.uuid4()),
                    context_id=context_id,
                    parts=[Part(text=pregunta)],
                )

                solicitud = SendMessageRequest(message=mensaje)

                # Recibir eventos emitidos por el servidor
                async for evento in client.send_message(solicitud):

                    if evento.HasField("artifact_update"):
                        artefacto = evento.artifact_update.artifact

                        texto = get_artifact_text(
                            artefacto,
                            delimiter=" ",
                        )

                        if texto:
                            print("Agente:", texto)

                    elif evento.HasField("status_update"):
                        actualizacion = evento.status_update
                        estado = TaskState.Name(
                            actualizacion.status.state
                        )

                        print("Estado:", estado)

                    elif evento.HasField("message"):
                        texto = get_message_text(
                            evento.message,
                            delimiter=" ",
                        )

                        if texto:
                            print("Agente:", texto)

        finally:
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
