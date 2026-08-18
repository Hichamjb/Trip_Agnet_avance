import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from backend.config import config
from backend.agent import build_trip_agent

# Configuration des logs professionnels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TripAgent")


# =====================================================================
# PYDANTIC SCHEMAS (avec exemples pour Swagger UI /docs)
# =====================================================================

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="Requête de voyage envoyée à l'agent AI.",
        json_schema_extra={
            "examples": [
                "Planifie un voyage de 3 jours à Paris avec la météo, les restaurants et les activités.",
                "Combien valent 500 EUR en USD ?",
                "Quel temps fait-il à Berlin actuellement ?"
            ]
        }
    )


class ChatResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Réponse finale structurée générée par l'agent de voyage.",
        json_schema_extra={
            "examples": [
                "Voici votre itinéraire de 3 jours à Paris :\n\n- Jour 1: Visite de la Tour Eiffel...\n- Météo: 22°C..."
            ]
        }
    )


class ToolArgumentDetail(BaseModel):
    type: str
    description: Optional[str] = None


class ToolSchema(BaseModel):
    name: str = Field(..., description="Nom de l'outil MCP.")
    description: str = Field(..., description="Description de la fonction de l'outil.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Paramètres acceptés par l'outil.")
    required_arguments: List[str] = Field(default_factory=list, description="Arguments obligatoires.")


class ToolsListResponse(BaseModel):
    total_tools: int
    tools: List[ToolSchema]


class HealthResponse(BaseModel):
    status: str
    agent: str


# =====================================================================
# LIFESPAN (Initialisation unique au démarrage)
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie FastAPI.
    Connecte le client MCP et instancie l'agent LangGraph une seule fois.
    """
    logger.info("Démarrage de Trip Agent Avance... Connexion aux serveurs MCP...")
    
    try:
        config.validate()
    except ValueError as err:
        logger.warning(f"Avertissement de configuration: {err}")

    try:
        agent, mcp_client = await build_trip_agent()
        app.state.agent = agent
        app.state.mcp_client = mcp_client
        logger.info("Trip Agent initialisé et prêt pour traiter les requêtes.")
    except Exception as e:
        logger.error(f"Échec critique d'initialisation de l'agent: {e}")
        app.state.agent = None
        app.state.mcp_client = None

    yield

    # Fermeture propre des connexions MCP à l'arrêt du serveur
    mcp_client = getattr(app.state, "mcp_client", None)
    if mcp_client:
        logger.info("Fermeture des connexions MCP...")
        if hasattr(mcp_client, "close"):
            await mcp_client.close()


# =====================================================================
# FASTAPI APP DEFENITION
# =====================================================================

app = FastAPI(
    title="Trip_Agnet_avance API",
    description="""
### API de l'Agent de Voyage Avancé AI (Multi-MCP Direct)

Cet agent utilise **LangGraph**, **Groq (llama-3.3-70b-versatile)** et **MultiServerMCPClient** pour exécuter dynamiquement des outils externes.

#### Fonctionnalités :
* **`/chat`** : Posez n'importe quelle question de voyage (Météo, recherche web, lieux, conversion, etc.).
* **`/tools`** : Inspectez tous les outils MCP découverts en temps réel.
    """,
    version="1.0.0",
    lifespan=lifespan
)


# =====================================================================
# ENDPOINTS
# =====================================================================

@app.get(
    "/",
    response_model=HealthResponse,
    tags=["Diagnostics"],
    summary="Vérification du statut de l'API"
)
async def get_status():
    """Retourne l'état de fonctionnement du serveur FastAPI."""
    return HealthResponse(
        status="running",
        agent="Trip Agent Advanced"
    )


@app.get(
    "/tools",
    response_model=ToolsListResponse,
    tags=["Tools & Diagnostics"],
    summary="Inspecter les outils MCP actifs"
)
async def list_mcp_tools():
    """
    **Endpoint d'inspection des outils MCP** :
    Découvre et liste tous les outils actuellement connectés via les serveurs MCP externes.
    """
    mcp_client = getattr(app.state, "mcp_client", None)
    if not mcp_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Le client MCP n'est pas initialisé ou a échoué au démarrage."
        )

    try:
        tools = await mcp_client.get_tools()
        formatted_tools = []

        for tool in tools:
            args_schema = getattr(tool, "args_schema", None)
            properties = {}
            required = []

            if args_schema:
                schema = args_schema.schema() if hasattr(args_schema, "schema") else {}
                properties = schema.get("properties", {})
                required = schema.get("required", [])

            formatted_tools.append(
                ToolSchema(
                    name=tool.name,
                    description=tool.description or "Pas de description disponible",
                    arguments=properties,
                    required_arguments=required
                )
            )

        return ToolsListResponse(
            total_tools=len(formatted_tools),
            tools=formatted_tools
        )
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des outils MCP: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'inspection des outils: {str(e)}"
        )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Travel Agent"],
    summary="Interagir avec l'agent de voyage AI"
)
async def chat_endpoint(request: ChatRequest):
    """
    **Endpoint de Chat principal** :
    Envoie une requête à l'agent AI. L'agent sélectionnera et exécutera automatiquement les outils MCP nécessaires pour répondre.
    """
    user_query = request.message.strip()
    if not user_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le message ne peut pas être vide."
        )

    agent = getattr(app.state, "agent", None)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="L'agent de voyage n'est pas initialisé. Vérifiez les logs du serveur."
        )

    try:
        inputs = {"messages": [HumanMessage(content=user_query)]}

        # Live Execution Trace dans la console de commande
        print("\n" + "=" * 70)
        print(f"👤 REQUÊTE UTILISATEUR: {user_query}")
        print("=" * 70)

        result = await agent.ainvoke(inputs)
        messages = result.get("messages", [])

        # Affichage du flux de décision du LLM et des appels MCP dans le terminal
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"\n🤖 DÉCISION LLM  → Appel de l'outil : [{tool_call['name']}]")
                    print(f"🛠️  Arguments      : {tool_call['args']}")

            elif isinstance(msg, ToolMessage):
                print(f"\n⚡ RÉSULTAT MCP   → Outil : [{msg.name}]")
                content_str = str(msg.content)
                preview = content_str[:250] + "..." if len(content_str) > 250 else content_str
                print(f"📦 Données        : {preview}")

        final_message = messages[-1]
        print("\n" + "-" * 70)
        print(f"🏁 RÉPONSE FINALE:\n{final_message.content}")
        print("=" * 70 + "\n")

        return ChatResponse(answer=str(final_message.content))

    except Exception as e:
        logger.error(f"Erreur d'exécution de la requête /chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur d'exécution du workflow Agent: {str(e)}"
        )