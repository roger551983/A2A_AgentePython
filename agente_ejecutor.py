import json
import logging
import os
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from openai import AsyncOpenAI
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from skills import AGENT_SKILLS, SKILL_DEFINITIONS

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / "variables_entorno.env", override=False)

#Variables de entorno

BIND_HOST = os.getenv("A2A_BIND_HOST", "0.0.0.0")
PORT = int(os.getenv("A2A_PORT", "9999"))
PUBLIC_SCHEME = os.getenv("A2A_PUBLIC_SCHEME", "http").rstrip(":")
PUBLIC_HOST = os.getenv("A2A_PUBLIC_HOST", "127.0.0.1")
PUBLIC_PORT = int(os.getenv("A2A_PUBLIC_PORT", str(PORT)))
RPC_PATH = os.getenv("A2A_RPC_PATH", "/a2a/jsonrpc")
EXECUTE_PATH = os.getenv("A2A_EXECUTE_PATH", "/tasks/execute")
CARD_PATH = os.getenv("A2A_AGENT_CARD_PATH", "/.well-known/agent-card.json")
AGENT_NAME = os.getenv(
    "A2A_AGENT_NAME", "Enterprise Travel & Emergency Specialist Agent"
)
AGENT_DESCRIPTION = os.getenv(
    "A2A_AGENT_DESCRIPTION",
    "Agente especializado en itinerarios, información de visados y orientación de salud durante viajes.",
)
AGENT_VERSION = os.getenv("A2A_AGENT_VERSION", "1.0.0")
MAX_HISTORY_MESSAGES = int(os.getenv("A2A_MAX_HISTORY_MESSAGES", "12"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))

PUBLIC_BASE_URL = f"{PUBLIC_SCHEME}://{PUBLIC_HOST}:{PUBLIC_PORT}"

logger = logging.getLogger("agente_a2a")

def get_skill_id(definition: dict[str, Any]) -> str:
    return str(definition.get("id") or definition.get("skill_id") or "").strip()

SKILLS_BY_ID: dict[str, dict[str, Any]] = {
    get_skill_id(item): item
    for item in SKILL_DEFINITIONS
    if isinstance(item, dict) and get_skill_id(item)
}


if not SKILLS_BY_ID:
    raise RuntimeError("SKILL_DEFINITIONS no contiene ninguna skill válida.")

def schema_for(skill_id: str) -> dict[str, Any]:
    """Busca el esquema dentro de SKILL_DEFINITIONS, sin otra constante."""
    requested_id = str(skill_id).strip()

    for definition in SKILL_DEFINITIONS:
        if not isinstance(definition, dict):
            continue
        current_id = str(definition.get("id") or definition.get("skill_id") or "").strip()
        if current_id != requested_id:
            continue

        # La versión del skills.py compartido usa parametersSchema.
        schema = definition.get("parametersSchema")
        if not isinstance(schema, dict):
            schema = definition.get("parameters_schema")
        if isinstance(schema, dict):
            return schema
        return {}

    return {}

def required_fields_for(skill_id: str) -> list[str]:
    schema = schema_for(skill_id)
    required = schema.get("required", [])
    if isinstance(required, list):
        return [str(value) for value in required]
    return []

def missing_required_fields(schema: dict[str, Any], data: dict[str, Any], prefix: str = "") -> list[str]:
    
    missing: list[str] = []
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    if not isinstance(required, list) or not isinstance(properties, dict):
        return missing
    for field in required:
        field_name = str(field)
        full_name = f"{prefix}.{field_name}" if prefix else field_name
        value = data.get(field_name) if isinstance(data, dict) else None
        if value is None or value == "" or value == []:
            missing.append(full_name)
            continue
        child_schema = properties.get(field_name, {})
        if isinstance(child_schema, dict) and child_schema.get("type") == "object":
            if not isinstance(value, dict):
                missing.append(full_name)
            else:
                missing.extend(missing_required_fields(child_schema, value, full_name))
    return missing

def skill_catalog_for_prompt() -> str:
    entries: list[dict[str, Any]] = []
    for skill_id, definition in SKILLS_BY_ID.items():
        entries.append(
            {
                "id": skill_id,
                "name": definition.get("name", skill_id),
                "description": definition.get("description", ""),
                "parameters_schema": schema_for(skill_id),
            }
        )
    return json.dumps(entries, ensure_ascii=False, indent=2)

SKILLS_CATALOG_JSON = skill_catalog_for_prompt()

openai_client: AsyncOpenAI | None = None
if OPENAI_API_KEY:
    openai_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=2,
    )
else:
    logger.warning(
        "OPENAI_API_KEY no está configurada. El servidor arrancará, pero las "
        "peticiones que necesiten IA responderán con HTTP 503."
    )

SYSTEM_PROMPT = f"""Eres {AGENT_NAME}.
Tu tarea es interpretar la solicitud y utilizar únicamente las skills que se
anuncian en el catálogo. No afirmes que consultaste fuentes externas si no lo
hiciste. En temas de visados, advierte que los requisitos deben verificarse con
la autoridad consular oficial. En consultas de salud, brinda orientación general,
no un diagnóstico; ante una emergencia real, recomienda contactar a los servicios
locales de emergencia.

Catálogo de skills disponible:
{SKILLS_CATALOG_JSON}

Debes responder SIEMPRE con un objeto JSON válido que contenga exactamente estas
claves principales:
{{
  "selected_skills": ["id_de_skill"],
  "missing_fields": [],
  "response": "respuesta clara en español"
}}
Si necesitas datos obligatorios que no fueron proporcionados, enuméralos en
missing_fields y formula una pregunta breve en response. No inventes los datos.
selected_skills solo puede incluir IDs del catálogo.
"""

def get_input_text(context: Any) -> str:
    message = getattr(context, "message", None)
    if message is None:
        return ""
    try:
        return get_message_text(message, delimiter=" ").strip()
    except TypeError:
        # Compatibilidad con versiones del helper sin el argumento delimiter.
        return str(get_message_text(message) or "").strip()
    except Exception:
        logger.exception("No se pudo extraer el texto del mensaje A2A.")
        return ""
def extract_json_object(content: str) -> dict[str, Any]:
    """Lee el JSON devuelto por OpenAI, con alternativa si viene delimitado."""
    content = (content or "").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("El modelo no devolvió un objeto JSON válido.")
        data = json.loads(content[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("La respuesta JSON del modelo no es un objeto.")
    return data

async def consultar_openai(
    solicitud: str,
    *,
    skill_forzada: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoca OpenAI y devuelve selected_skills, missing_fields y response."""
    if openai_client is None:
        raise RuntimeError("Falta configurar OPENAI_API_KEY en el archivo de entorno.")

    payload: dict[str, Any] = {"request": solicitud}
    if skill_forzada:
        payload["requested_skill"] = skill_forzada
        payload["parameters"] = parameters or {}
        payload["parameters_schema"] = schema_for(skill_forzada)
        payload["required_fields"] = required_fields_for(skill_forzada)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Procesa la siguiente solicitud. Devuelve exclusivamente el JSON "
                "indicado por las instrucciones del sistema:\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            ),
        },
    ]

    if skill_forzada:
        # En el endpoint /tasks/execute se respeta operation y no se permite
        # que el modelo elija otra skill.
        messages.append(
            {
                "role": "system",
                "content": (
                    f"La skill solicitada por el sistema llamador es '{skill_forzada}'. "
                    "Usa exclusivamente esta skill. Evalúa los campos obligatorios "
                    "con el esquema y los parámetros proporcionados. Si faltan campos, "
                    "indícalos en missing_fields y pide esos datos en response."
                ),
            }
        )

    completion = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=OPENAI_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content or "{}"
    result = extract_json_object(content)

    selected = result.get("selected_skills", [])
    if not isinstance(selected, list):
        selected = []
    selected = [str(item) for item in selected if str(item) in SKILLS_BY_ID]

    if skill_forzada:
        selected = [skill_forzada]

    missing_fields = result.get("missing_fields", [])
    if not isinstance(missing_fields, list):
        missing_fields = []

    answer = result.get("response", "")
    if not isinstance(answer, str):
        answer = json.dumps(answer, ensure_ascii=False)
    if not answer.strip():
        answer = "No pude generar una respuesta. Reformula la solicitud, por favor."

    return {
        "selected_skills": selected,
        "missing_fields": [str(value) for value in missing_fields],
        "response": answer.strip(),
    }

async def ejecutar_toolcall(request: Request) -> JSONResponse:
    """Endpoint sencillo compatible con llamadas tipo toolCall desde Postman."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "El cuerpo debe ser JSON válido."}, status_code=400
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "El cuerpo JSON debe ser un objeto."}, status_code=400
        )

    tool_call = body.get("toolCall", body)
    if not isinstance(tool_call, dict):
        return JSONResponse(
            {"error": "toolCall debe ser un objeto JSON."}, status_code=400
        )

    operation = str(tool_call.get("operation", "")).strip()
    parameters = tool_call.get("parameters", tool_call.get("params", {}))
    if not operation:
        return JSONResponse(
            {"error": "Falta toolCall.operation. Ejemplo: 'visado'."},
            status_code=400,
        )
    if operation not in SKILLS_BY_ID:
        return JSONResponse(
            {
                "error": f"La operación '{operation}' no existe.",
                "operaciones_validas": sorted(SKILLS_BY_ID.keys()),
            },
            status_code=400,
        )
    if not isinstance(parameters, dict):
        return JSONResponse(
            {"error": "toolCall.parameters debe ser un objeto JSON."},
            status_code=400,
        )

    missing = missing_required_fields(schema_for(operation), parameters)
    if missing:
        questions = ", ".join(missing)
        return JSONResponse(
            {
                "result": f"Para ejecutar la skill '{operation}', proporciona estos campos obligatorios: {questions}.",
                "skillUtilizado": operation,
                "missing_fields": missing,
            },
            status_code=200,
        )

    try:
        request_text = (
            f"Ejecuta la skill '{operation}' con los parámetros recibidos. "
            "Responde en español y utiliza esos valores como contexto."
        )
        result = await consultar_openai(
            request_text,
            skill_forzada=operation,
            parameters=parameters,
        )
        return JSONResponse(
            {
                "result": result["response"],
                "skillUtilizado": operation,
                "selected_skills": result["selected_skills"],
                "missing_fields": result["missing_fields"],
            },
            status_code=200,
        )
    except RuntimeError as exc:
        logger.error("Configuración OpenAI: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception:
        logger.exception("Error ejecutando toolCall '%s'", operation)
        return JSONResponse(
            {"error": "Ocurrió un error al ejecutar la skill. Revisa los logs del servidor."},
            status_code=500,
        )

from a2a.helpers import get_message_text, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, Part

class TravelAgentExecutor(AgentExecutor):
    """Ejecuta solicitudes por A2A JSON-RPC usando las skills de skills.py."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = getattr(context, "message", None)
        if message is None:
            logger.warning("Se recibió una solicitud A2A sin mensaje.")
            return

        task = getattr(context, "current_task", None) or new_task_from_user_message(message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )

        try:
            await updater.start_work(
                updater.new_agent_message(
                    parts=[Part(text="Analizando la solicitud y las skills disponibles...")]
                )
            )

            user_text = get_input_text(context)
            if not user_text:
                answer = "Envía una solicitud en texto para que pueda seleccionar una skill."
                output = {"result": answer, "skillUtilizado": None, "missing_fields": []}
            else:
                result = await consultar_openai(user_text)
                selected = result["selected_skills"]
                answer = result["response"]
                if selected:
                    answer = f"Skills seleccionadas: {', '.join(selected)}\n\n{answer}"
                output = {
                    "result": answer,
                    "skillUtilizado": selected[0] if selected else None,
                    "selected_skills": selected,
                    "missing_fields": result["missing_fields"],
                }

            response_json = json.dumps(output, ensure_ascii=False)
            await updater.add_artifact(
                parts=[Part(text=response_json)],
                name="respuesta",
                last_chunk=True,
            )
            await updater.complete(
                updater.new_agent_message(parts=[Part(text="Solicitud completada.")])
            )
        except Exception:
            logger.exception("Falló el procesamiento de la tarea A2A.")
            try:
                await updater.failed(
                    updater.new_agent_message(
                        parts=[Part(text="No se pudo procesar la solicitud. Revisa los logs del servidor.")]
                    )
                )
            except Exception:
                logger.exception("No fue posible publicar el estado de error A2A.")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("La cancelación de tareas no está implementada.")

AGENT_CARD = AgentCard(
    name=AGENT_NAME,
    description=AGENT_DESCRIPTION,
    version=AGENT_VERSION,
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[
        AgentInterface(
        protocol_binding="JSONRPC",
        protocol_version="0.3",
        url=f"{PUBLIC_BASE_URL}{RPC_PATH}",
    ),
        AgentInterface(
            protocol_binding="JSONRPC",
            protocol_version="1.0",
            url=f"{PUBLIC_BASE_URL}{RPC_PATH}",
        )
    ],
    skills=AGENT_SKILLS,
)

request_handler = DefaultRequestHandler(
        agent_executor=TravelAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=AGENT_CARD,
    )

async def health(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "agent": AGENT_NAME,
            "model": OPENAI_MODEL,
            "openai_configured": bool(OPENAI_API_KEY),
            "skills": sorted(SKILLS_BY_ID.keys()),
            "rpc_url": f"{PUBLIC_BASE_URL}{RPC_PATH}",
            "agent_card_url": f"{PUBLIC_BASE_URL}/.well-known/agent-card.json",
        }
    )


routes = [
    Route("/health", health, methods=["GET"]),
    Route(EXECUTE_PATH, ejecutar_toolcall, methods=["POST"]),
]
routes.extend(create_agent_card_routes(AGENT_CARD, card_url=CARD_PATH))
routes.extend(create_jsonrpc_routes(request_handler, rpc_url=RPC_PATH))
routes.extend(create_jsonrpc_routes(request_handler,rpc_url=RPC_PATH,enable_v0_3_compat=True))
routes.extend(create_agent_card_routes(AGENT_CARD,card_url="/a2a/.well-known/agent-card.json"))
app = Starlette(routes=routes)


def main() -> None:
    logger.info("Agente A2A: %s", AGENT_NAME)
    logger.info("Escuchando en http://%s:%s", BIND_HOST, PORT)
    logger.info("Health:       http://%s:%s/health", PUBLIC_HOST, PUBLIC_PORT)
    logger.info("Agent Card:   %s/.well-known/agent-card.json", PUBLIC_BASE_URL)
    logger.info("A2A JSON-RPC: %s%s", PUBLIC_BASE_URL, RPC_PATH)
    logger.info("Tool endpoint:%s%s", PUBLIC_BASE_URL, EXECUTE_PATH)
    logger.info("Skills: %s", ", ".join(sorted(SKILLS_BY_ID.keys())))
    uvicorn.run(app, host=BIND_HOST, port=PORT, log_level=LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
