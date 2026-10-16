from a2a.types import AgentSkill

SKILL_DEFINITIONS = [

  {
        "id": "itinerario",
        "name": "Planificación de itinerarios",
        "description": (
            "Planifica viajes e itinerarios sostenibles, "
            "recomienda alternativas de transporte y alojamiento "
            "y organiza las actividades según las preferencias "
            "del viajero."
        ),
        "tags": [
            "viaje",
            "itinerario",
            "turismo",
            "sostenibilidad",
        ],
        "examples": [
            "Planifica un viaje sostenible de Bogotá a Buenos Aires.",
            "Organiza un itinerario turístico de cinco días.",
        ],
        "inputModes": ["text"],
        "outputModes": ["text"],
        "parametersSchema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "Solicitud y preferencias del viaje",
                },
                "geoCoordinates": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                    },
                    "required": ["latitude", "longitude"],
                },
                "temperature": {"type": "number"},
            },
            "required": [
                "request",
                "geoCoordinates",
                "temperature",
            ],
        },
    },
    {
        "id": "visado",
        "name": "Visados y requisitos de entrada",
        "description": (
            "Orienta sobre requisitos de visado, documentación "
            "migratoria, pasaportes y condiciones de entrada "
            "a destinos internacionales."
        ),
        "tags": [
            "visado",
            "visa",
            "pasaporte",
            "inmigracion",
            "requisitos_entrada",
        ],
        "examples": [
            "¿Qué requisitos de visado tiene un ciudadano peruano "
            "que viaja a Japón?",
            "¿Qué documentos necesito para viajar a España?",
        ],
        "inputModes": ["text"],
        "outputModes": ["text"],
        "parametersSchema": {
            "type": "object",
            "properties": {
                "originCountry": {"type": "string"},
                "destinationCountry": {"type": "string"},
                "passportType": {"type": "string"},
            },
            "required": [
                "originCountry",
                "destinationCountry",
            ],
        },
    },
    {
        "id": "salud",
        "name": "Asistencia sanitaria durante viajes",
        "description": (
            "Ofrece orientación general ante problemas de salud "
            "durante un viaje, ayuda a identificar necesidades "
            "de asistencia médica y considera la ubicación "
            "y el seguro de viaje cuando corresponda."
        ),
        "tags": [
            "salud",
            "medicina",
            "emergencia",
            "hospital",
            "seguro_viaje",
        ],
        "examples": [
            "Necesito orientación médica urgente durante mi viaje.",
            "¿Cómo puedo encontrar asistencia sanitaria en Madrid?",
        ],
        "inputModes": ["text"],
        "outputModes": ["text"],
        "parametersSchema": {
            "type": "object",
            "properties": {
                "emergencyType": {"type": "string"},
                "geoCoordinates": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                    },
                    "required": ["latitude", "longitude"],
                },
                "insurancePolicyId": {"type": "string"},
            },
            "required": [
                "request"
            ],
        },
    },
]



AGENT_SKILLS = [
    AgentSkill(
        id=definition["id"],
        name=definition["name"],
        description=definition["description"],
        tags=definition["tags"],
        examples=definition["examples"],
        input_modes=definition["inputModes"],
        output_modes=definition["outputModes"],
    )
    for definition in SKILL_DEFINITIONS
]
