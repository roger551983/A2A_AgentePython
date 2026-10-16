import json
import logging
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from openai import AsyncOpenAI
from starlette.applications import Starlette

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_artifact_update_event,
    new_text_status_update_event,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    TaskState,
)

from skills import AGENT_SKILLS, SKILL_DEFINITIONS

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
if not ENV_FILE.is_file():
    ENV_FILE = BASE_DIR / "variables_entorno.env"

if ENV_FILE.is_file():
    # Docker Compose may already inject the variables; don't overwrite them.
    load_dotenv(dotenv_path=ENV_FILE, override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))

if not OPENAI_API_KEY or OPENAI_API_KEY == "PEGA_AQUI_TU_CLAVE_REAL":
    raise RuntimeError(
        f"Configura OPENAI_API_KEY en {ENV_FILE} o en las variables de Docker."
    )

openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    timeout=OPENAI_TIMEOUT_SECONDS,
)

BIND_HOST = os.getenv("A2A_BIND_HOST", "0.0.0.0")
PORT = int(os.getenv("A2A_PORT", "9999"))

PUBLIC_SCHEME = os.getenv("A2A_PUBLIC_SCHEME", "http")
PUBLIC_HOST = os.getenv("A2A_PUBLIC_HOST", "127.0.0.1")
PUBLIC_PORT = int(os.getenv("A2A_PUBLIC_PORT", str(PORT)))
RPC_PATH = os.getenv("A2A_RPC_PATH", "/a2a/jsonrpc")
if not RPC_PATH.startswith("/"):
    RPC_PATH = "/" + RPC_PATH

PUBLIC_RPC_URL = f"{PUBLIC_SCHEME}://{PUBLIC_HOST}:{PUBLIC_PORT}{RPC_PATH}"


AGENT_NAME = os.getenv(
    "A2A_AGENT_NAME",
    "Enterprise Travel & Emergency Specialist Agent",
)
AGENT_DESCRIPTION = os.getenv(
    "A2A_AGENT_DESCRIPTION",
    "Agente A2A para viajes sostenibles, visados, asistencia durante viajes "
    "y logística de turismo espacial.",
)
AGENT_VERSION = os.getenv("A2A_AGENT_VERSION", "1.0.0")
AGENT_CARD_PATH = os.getenv(
    "A2A_AGENT_CARD_PATH",
    "/.well-known/agent-card.json",
)
MAX_HISTORY_MESSAGES = max(2, int(os.getenv("A2A_MAX_HISTORY_MESSAGES", "12")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("a2a_travel_agent")

def construir_prompt_dinamico(historial: list[dict], pregunta: str) -> list[dict]:
    catalogo_json = json.dumps(
        SKILL_DEFINITIONS,
        ensure_ascii=False,
        indent=2,
    )

    system_prompt = f"""
    Eres un agente inteligente especializado en viajes y asistencia.

    IDENTIDAD:
    Nombre: {AGENT_NAME}
    Descripción: {AGENT_DESCRIPTION}

    CATÁLOGO DE SKILLS, EJEMPLOS Y ESQUEMAS:
    {catalogo_json}

    INSTRUCCIONES:
    1. Interpreta la solicitud escrita en lenguaje natural.
    2. Selecciona el skill más pertinente usando id, descripción, tags,
    ejemplos y parametersSchema.
    3. Considera el historial para entender referencias anteriores.
    4. Si faltan datos importantes, pregunta por ellos.
    5. Responde en español, salvo que el usuario solicite otro idioma.
    6. No afirmes que realizaste reservas, verificaciones oficiales,
    localización de hospitales o activación de seguros si no se ejecutó
    realmente una herramienta o servicio externo.
    7. Los skills en el catálogo describen capacidades; este ejemplo aún
    no ejecuta servicios externos asociados a ellos.
    8. Selecciona uno o más skills válidos, priorizando un skill principal.
    9. Devuelve únicamente un objeto JSON válido con esta estructura:
    {{
    "selected_skills": ["id_del_skill"],
    "missing_fields": [],
    "response": "Respuesta natural para el usuario"
    }}
    10. selected_skills debe contener IDs exactos del catálogo.
    11. missing_fields debe ser una lista de datos que falten.
    12. response debe contener la respuesta natural para el usuario.
    13 Para consultas generales de salud durante un viaje, no solicites
    coordenadas si el usuario ya indicó el lugar. Pregunta por la
    ubicación precisa únicamente cuando una herramienta realmente
    la necesite para ejecutar una operación.
    No escribas Markdown fuera del JSON.
    """

    return [
        {"role": "system", "content": system_prompt},
        *historial,
        {"role": "user", "content": pregunta},
    ]

class TravelAgentExecutor(AgentExecutor):
     def __init__(self) -> None:
        self.historiales: dict[str, list[dict]] = {}
        self.skill_ids_validos = {
            item["id"] for item in SKILL_DEFINITIONS
        }
     async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

         if context.message is None:
            logger.warning("Solicitud A2A sin mensaje.")
            return
         task = context.current_task
         if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

         try:
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=TaskState.TASK_STATE_WORKING,
                    text="Analizando la solicitud y los skills...",
                )
            )

            pregunta = get_message_text(context.message).strip()
            if not pregunta:
                await event_queue.enqueue_event(
                    new_text_status_update_event(
                        task_id=task.id,
                        context_id=task.context_id,
                        state=TaskState.TASK_STATE_FAILED,
                        text="No se recibió texto en la solicitud.",
                    )
                )
                return

            context_id = task.context_id
            historial = self.historiales.setdefault(context_id, [])
            mensajes = construir_prompt_dinamico(historial, pregunta)

            completion = await openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=mensajes,
                temperature=OPENAI_TEMPERATURE,
                response_format={"type": "json_object"},
            )

            contenido = completion.choices[0].message.content or "{}"
            try:
                resultado = json.loads(contenido)
            except json.JSONDecodeError:
                logger.warning("OpenAI devolvió JSON inválido; se usará el texto recibido.")
                resultado = {
                    "selected_skills": [],
                    "missing_fields": [],
                    "response": contenido,
                }
            respuesta = resultado.get("response", "No se obtuvo una respuesta.")
            if not isinstance(respuesta, str) or not respuesta.strip():
                respuesta = "No se obtuvo una respuesta válida."

            selected_skills = resultado.get("selected_skills", [])
            if not isinstance(selected_skills, list):
                selected_skills = []

            selected_skills = [
                skill_id
                for skill_id in selected_skills
                if isinstance(skill_id, str) and skill_id in self.skill_ids_validos
            ]

            skill_utilizado = selected_skills[0] if selected_skills else None

            # Final format required by the user / Camunda mapping.
            response_data = {
                "result": respuesta,
                "skillUtilizado": skill_utilizado,
            }
            response_json = json.dumps(response_data, ensure_ascii=False)

            logger.info("Contexto A2A: %s", context_id)
            logger.info("Skill utilizado: %s", skill_utilizado)

            historial.extend([
                {"role": "user", "content": pregunta},
                {"role": "assistant", "content": respuesta},
            ])
            del historial[:-MAX_HISTORY_MESSAGES]

            await event_queue.enqueue_event(
                new_text_artifact_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    name="respuesta",
                    text=response_json,
                )
            )

            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=TaskState.TASK_STATE_COMPLETED,
                    text="Consulta completada.",
                )
            )
         except:

             logger.exception("Error procesando la tarea A2A.")
             await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=TaskState.TASK_STATE_FAILED,
                    text=(
                        "No se pudo procesar la solicitud. "
                        "Revisa la configuración de OpenAI y los logs."
                    ),
                )
            )

     async def cancel(
         self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancellation is not implemented in this example."""
        raise NotImplementedError("La cancelación de tareas no está implementada.")
agent_card = AgentCard(
        name=AGENT_NAME,
        description=AGENT_DESCRIPTION,
        version=AGENT_VERSION,
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=PUBLIC_RPC_URL,
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=AGENT_SKILLS,
    )

request_handler = DefaultRequestHandler(
    agent_executor=TravelAgentExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)



routes = []
routes.extend(create_agent_card_routes(agent_card))
routes.extend(create_jsonrpc_routes(request_handler, rpc_url=RPC_PATH))

app = Starlette(routes=routes)

if __name__ == "__main__":
    logger.info("Servidor escuchando en %s:%s", BIND_HOST, PORT)
    logger.info("PUBLIC_RPC_URL=%s", PUBLIC_RPC_URL)
    logger.info("Agent Card: %s://%s:%s/.well-known/agent-card.json", PUBLIC_SCHEME, PUBLIC_HOST, PUBLIC_PORT)
    uvicorn.run(app, host=BIND_HOST, port=PORT)
