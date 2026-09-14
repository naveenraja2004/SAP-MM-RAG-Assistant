
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SAP MM RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("SAP MM RAG Assistant 🤖")
st.caption("SAP MM Knowledge + RAG + T-Codes + Transaction Guides + Troubleshooting")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge" / "sap_mm_knowledge.txt"

# ============================================================
# T-CODE DATABASE
# ============================================================

TCODE_DATABASE = {
    "ME51N": "Create Purchase Requisition",
    "ME52N": "Change Purchase Requisition",
    "ME53N": "Display Purchase Requisition",
    "ME54N": "Release Purchase Requisition",
    "ME55": "Collective Release of Purchase Requisitions",

    "ME41": "Create Request for Quotation",
    "ME42": "Change Request for Quotation",
    "ME43": "Display Request for Quotation",

    "ME21N": "Create Purchase Order",
    "ME22N": "Change Purchase Order",
    "ME23N": "Display Purchase Order",

    "MIGO": "Goods Movement / Goods Receipt",
    "MIRO": "Enter Incoming Invoice",

    "MM01": "Create Material",
    "MM02": "Change Material",
    "MM03": "Display Material",

    "ME01": "Maintain Source List",

    "ME11": "Create Purchasing Info Record",
    "ME12": "Change Purchasing Info Record",
    "ME13": "Display Purchasing Info Record",

    "ME31K": "Create Contract",
    "ME32K": "Change Contract",
    "ME33K": "Display Contract",

    "ME31L": "Create Scheduling Agreement",
    "ME32L": "Change Scheduling Agreement",
    "ME33L": "Display Scheduling Agreement",

    "MMBE": "Stock Overview",
    "MD04": "Stock / Requirements List",
    "MD01N": "MRP Run",

    "OBYC": "Automatic Account Determination",
    "BP": "Business Partner"
}

# ============================================================
# MOVEMENT TYPES
# ============================================================

MOVEMENT_TYPES = {
    "101": "Goods Receipt for Purchase Order",
    "102": "Reversal of Goods Receipt",
    "201": "Goods Issue to Cost Center",
    "202": "Reversal of 201",
    "261": "Goods Issue to Production Order",
    "262": "Reversal of 261",
    "301": "Transfer Posting Plant to Plant",
    "311": "Storage Location to Storage Location Transfer",
    "351": "Goods Issue for Stock Transport Order",
    "641": "Goods Issue for Stock Transport",
    "501": "Goods Receipt Without Purchase Order"
}

# ============================================================
# TOPIC -> T-CODE MAP
# ============================================================

TOPIC_TCODE_MAP = {
    "purchase requisition": ["ME51N", "ME52N", "ME53N"],
    "pr": ["ME51N", "ME52N", "ME53N"],
    "rfq": ["ME41", "ME42", "ME43"],
    "request for quotation": ["ME41", "ME42", "ME43"],
    "purchase order": ["ME21N", "ME22N", "ME23N"],
    "po": ["ME21N", "ME22N", "ME23N"],
    "goods receipt": ["MIGO"],
    "migo": ["MIGO"],
    "invoice": ["MIRO"],
    "miro": ["MIRO"],
    "source list": ["ME01"],
    "info record": ["ME11", "ME12", "ME13"],
    "material master": ["MM01", "MM02", "MM03"],
    "contract": ["ME31K", "ME32K", "ME33K"],
    "scheduling agreement": ["ME31L", "ME32L", "ME33L"],
    "stock": ["MMBE", "MD04"],
    "mrp": ["MD01N"],
    "account determination": ["OBYC"],
    "business partner": ["BP"],
    "vendor": ["BP"]
}

# ============================================================
# TRANSACTION GUIDES
# ============================================================

TRANSACTION_GUIDES = {

    "ME51N": {
        "title": "Create Purchase Requisition",
        "purpose": "ME51N is used to create a Purchase Requisition (PR) for requesting materials or services.",
        "steps": [
            "Enter transaction code ME51N.",
            "Select the required PR document type if applicable.",
            "Enter the required material or short text.",
            "Enter the required quantity.",
            "Enter the delivery date.",
            "Enter the Plant.",
            "Enter the Storage Location if required.",
            "Enter Purchasing Group if required.",
            "Enter account assignment for consumable items if required.",
            "Check all mandatory fields.",
            "Click Save.",
            "SAP generates a Purchase Requisition number."
        ],
        "example": [
            "Material: RM-1001",
            "Quantity: 100",
            "Plant: Chennai Plant",
            "Delivery Date: Required date"
        ],
        "related": ["ME51N", "ME52N", "ME53N"]
    },

    "ME41": {
        "title": "Create Request for Quotation",
        "purpose": "ME41 is used to create an RFQ and request quotations from vendors.",
        "steps": [
            "Enter transaction code ME41.",
            "Enter RFQ document type if required.",
            "Enter the RFQ validity or required dates.",
            "Enter the Purchasing Organization.",
            "Enter the Purchasing Group.",
            "Enter the Plant.",
            "Enter the material or short text.",
            "Enter the required quantity.",
            "Enter delivery date.",
            "Enter vendor information when applicable.",
            "Check the RFQ data.",
            "Save the RFQ."
        ],
        "example": [
            "Material: RM-1001",
            "Quantity: 500",
            "Plant: Chennai Plant",
            "RFQ sent to multiple suppliers for quotation"
        ],
        "related": ["ME41", "ME42", "ME43"]
    },

    "ME21N": {
        "title": "Create Purchase Order",
        "purpose": "ME21N is used to create a Purchase Order (PO) for purchasing materials or services from a supplier.",
        "steps": [
            "Enter transaction code ME21N.",
            "Select the required Purchase Order document type.",
            "Enter the Vendor number.",
            "Enter the Purchasing Organization.",
            "Enter the Purchasing Group.",
            "Enter the Company Code.",
            "In Item Overview, enter the Material number.",
            "Enter the required PO quantity.",
            "Enter the Plant.",
            "Enter Storage Location if required.",
            "Check the item details and delivery information.",
            "Check pricing and conditions.",
            "Verify all mandatory fields.",
            "Click Save.",
            "SAP generates a Purchase Order number."
        ],
        "example": [
            "Vendor: ABC Suppliers",
            "Material: RM-1001",
            "Quantity: 100",
            "Plant: Chennai Plant"
        ],
        "related": ["ME21N", "ME22N", "ME23N"]
    },

    "MIGO": {
        "title": "Goods Receipt",
        "purpose": "MIGO is used to perform goods movements such as Goods Receipt against a Purchase Order.",
        "steps": [
            "Enter transaction code MIGO.",
            "Select Goods Receipt.",
            "Select Reference as Purchase Order.",
            "Enter the Purchase Order number.",
            "Press Enter.",
            "Verify material and quantity.",
            "Enter or verify the Plant.",
            "Check Storage Location.",
            "Select the Item OK indicator.",
            "Check the quantity received.",
            "Check the document details.",
            "Click Post.",
            "SAP generates a Material Document number."
        ],
        "example": [
            "PO: 4500001234",
            "Material: RM-1001",
            "Received Quantity: 100",
            "Movement Type: 101"
        ],
        "related": ["MIGO", "MB03", "MMBE"]
    },

    "MIRO": {
        "title": "Invoice Verification",
        "purpose": "MIRO is used to enter and post a vendor invoice against a Purchase Order and Goods Receipt.",
        "steps": [
            "Enter transaction code MIRO.",
            "Select Invoice transaction.",
            "Enter the invoice date.",
            "Enter the posting date.",
            "Enter the vendor invoice reference number.",
            "Enter the invoice amount.",
            "Enter the Purchase Order number.",
            "Check the proposed invoice items.",
            "Verify quantity and amount.",
            "Check tax information if required.",
            "Check the balance.",
            "Post the invoice.",
            "SAP generates an invoice document."
        ],
        "example": [
            "Vendor: ABC Suppliers",
            "PO: 4500001234",
            "Invoice Amount: â‚¹50,000",
            "Invoice posted after verification"
        ],
        "related": ["MIRO", "MIGO", "ME23N"]
    },

    "ME01": {
        "title": "Maintain Source List",
        "purpose": "ME01 is used to maintain valid sources of supply for a material and plant.",
        "steps": [
            "Enter transaction code ME01.",
            "Enter the Material number.",
            "Enter the Plant.",
            "Press Enter.",
            "Enter the Vendor as source of supply.",
            "Enter validity period.",
            "Set Fixed Source indicator if required.",
            "Set Block indicator if required.",
            "Check the source list entry.",
            "Save."
        ],
        "example": [
            "Material: RM-1001",
            "Plant: Chennai Plant",
            "Vendor: ABC Suppliers",
            "Validity: Current purchasing period"
        ],
        "related": ["ME01", "ME03", "ME11"]
    },

    "ME11": {
        "title": "Create Purchasing Info Record",
        "purpose": "ME11 is used to create a Purchasing Info Record containing vendor-material purchasing information.",
        "steps": [
            "Enter transaction code ME11.",
            "Enter Vendor.",
            "Enter Material.",
            "Enter Purchasing Organization.",
            "Enter Plant if applicable.",
            "Select the required Info Category.",
            "Enter vendor-specific purchasing data.",
            "Enter planned delivery time if required.",
            "Enter price or conditions if required.",
            "Check the data.",
            "Save."
        ],
        "example": [
            "Vendor: ABC Suppliers",
            "Material: RM-1001",
            "Purchasing Organization: 1000",
            "Vendor Material Price maintained"
        ],
        "related": ["ME11", "ME12", "ME13"]
    },

    "MM01": {
        "title": "Create Material Master",
        "purpose": "MM01 is used to create a new material master record.",
        "steps": [
            "Enter transaction code MM01.",
            "Enter the Material number if external numbering is used.",
            "Select the required Industry Sector.",
            "Select Material Type.",
            "Select the required views.",
            "Enter Organizational Levels such as Plant and Storage Location.",
            "Maintain Basic Data.",
            "Maintain Purchasing data.",
            "Maintain MRP data if required.",
            "Maintain Accounting data if required.",
            "Check all mandatory fields.",
            "Save the material."
        ],
        "example": [
            "Material: RM-1001",
            "Material Type: ROH",
            "Plant: Chennai Plant",
            "Material description: Raw Material"
        ],
        "related": ["MM01", "MM02", "MM03"]
    },

    "BP": {
        "title": "Create / Maintain Business Partner",
        "purpose": "BP is used in SAP S/4HANA to create and maintain Business Partner master data, including supplier-related roles.",
        "steps": [
            "Enter transaction code BP.",
            "Select the required Business Partner category.",
            "Enter the Business Partner name.",
            "Maintain address details.",
            "Maintain communication information.",
            "Assign the required Business Partner role.",
            "Maintain supplier-related purchasing data if applicable.",
            "Maintain Company Code data if applicable.",
            "Maintain Purchasing Organization data if applicable.",
            "Check mandatory fields.",
            "Save the Business Partner."
        ],
        "example": [
            "Business Partner: ABC Suppliers",
            "Role: Supplier / Vendor-related role",
            "Purchasing Organization: 1000",
            "Company Code: 1000"
        ],
        "related": ["BP", "ME21N", "ME11"]
    },

    "OBYC": {
        "title": "Automatic Account Determination",
        "purpose": "OBYC is used to configure automatic G/L account determination for inventory and related MM transactions.",
        "steps": [
            "Enter transaction code OBYC.",
            "Select the required Chart of Accounts.",
            "Select the relevant transaction key.",
            "Check the valuation class and account determination settings.",
            "Maintain the required G/L account where configuration permits.",
            "Check valuation grouping and related configuration.",
            "Save the configuration.",
            "Test the relevant MM transaction."
        ],
        "example": [
            "Transaction: Goods Receipt",
            "Movement Type: 101",
            "SAP determines inventory and GR/IR-related accounts automatically based on configuration."
        ],
        "related": ["OBYC", "MIGO", "MIRO"]
    }
}

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# KNOWLEDGE BASE
# ============================================================

@st.cache_data
def load_knowledge():
    if not KNOWLEDGE_FILE.exists():
        return []

    text = KNOWLEDGE_FILE.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    chunks = [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n", text)
        if chunk.strip()
    ]

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_data
def create_embeddings(_model, chunks):
    if not chunks:
        return np.empty((0, 384))

    embeddings = _model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.array(embeddings)


# ============================================================
# RAG SEARCH
# ============================================================

def rag_search(question, model, chunks, embeddings, top_k=3):

    if not chunks or embeddings.size == 0:
        return []

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    scores = np.dot(embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "text": chunks[idx],
            "score": float(scores[idx])
        })

    return results


# ============================================================
# MOVEMENT TYPE
# ============================================================

def detect_movement_type(question):

    for movement_type, description in MOVEMENT_TYPES.items():

        if re.search(
            rf"\b{re.escape(movement_type)}\b",
            question,
            re.IGNORECASE
        ):
            return movement_type, description

    return None, None


# ============================================================
# STRUCTURED TOPIC ANSWERS
# ============================================================

def get_topic_answer(question):

    q = question.lower()

    if "what is sap mm" in q or q.strip() == "sap mm":
        return (
            "SAP MM stands for Materials Management. "
            "It manages procurement, purchasing, inventory, "
            "material master and related business processes."
        )

    if "what is purchase requisition" in q or "what is pr" in q:
        return (
            "Purchase Requisition (PR) is an internal request "
            "to purchase materials or services."
        )

    if "what is rfq" in q or "request for quotation" in q:
        return (
            "RFQ stands for Request for Quotation. "
            "It is used to request price and delivery quotations "
            "from vendors."
        )

    if "what is purchase order" in q or "what is po" in q:
        return (
            "Purchase Order (PO) is a formal purchasing document "
            "sent to a vendor specifying material/service, quantity, "
            "price and delivery information."
        )

    if "what is migo" in q:
        return (
            "MIGO is used for goods movements such as Goods Receipt "
            "against a Purchase Order."
        )

    if "what is miro" in q:
        return (
            "MIRO is used for invoice verification and posting "
            "of vendor invoices."
        )

    if "source list" in q:
        return (
            "Source List contains valid sources of supply for a "
            "material and plant."
        )

    if "info record" in q:
        return (
            "Purchasing Info Record stores vendor-material purchasing "
            "information such as price and delivery details."
        )

    if "contract" in q:
        return (
            "A Contract is an outline agreement with a vendor "
            "for agreed purchasing conditions."
        )

    if "scheduling agreement" in q:
        return (
            "Scheduling Agreement is a long-term purchasing agreement "
            "with delivery schedules."
        )

    if "material master" in q:
        return (
            "Material Master stores central information about materials "
            "used across purchasing, inventory, MRP and accounting."
        )

    if "obyc" in q or "account determination" in q:
        return (
            "OBYC is used for automatic G/L account determination "
            "in SAP MM."
        )

    if "business partner" in q or q.strip() == "bp":
        return (
            "Business Partner is used in SAP S/4HANA to maintain "
            "business partner and supplier/customer-related master data."
        )

    if "p2p" in q or "procure to pay" in q:
        return (
        "SAP MM Procure-to-Pay (P2P) is the complete purchasing process. "
        "The main flow is: PR â†’ PO â†’ GR â†’ MIRO. "
        "1. PR (ME51N): Create a Purchase Requisition for the required material or service. "
        "2. PO (ME21N): Create a Purchase Order and send it to the selected vendor. "
        "3. GR (MIGO): Receive the material against the PO, normally using movement type 101. "
        "4. MIRO: Verify and post the vendor invoice against the purchasing documents."
    )

    return None


# ============================================================
# QUESTION T-CODES
# ============================================================

def get_question_tcodes(question):

    q = question.lower()

    found = []

    for tcode, description in TCODE_DATABASE.items():

        if re.search(
            rf"\b{re.escape(tcode.lower())}\b",
            q
        ):
            found.append((tcode, description))

    for topic, tcodes in TOPIC_TCODE_MAP.items():

        if topic in q:

            for tcode in tcodes:

                item = (tcode, TCODE_DATABASE.get(tcode, ""))

                if item not in found:
                    found.append(item)

    return found


def display_tcodes(tcodes):

    if not tcodes:
        return

    st.markdown("### 📌 Related T-Codes")

    for tcode, description in tcodes:
        st.markdown(f"- `{tcode}` {chr(0x2014)} {description}")


# ============================================================
# PRACTICAL EXAMPLE
# ============================================================

def generate_practical_example(question):

    q = question.lower()

    if "release strategy" in q:
        return (
            "Example: A PR above a predefined value requires manager approval. "
            "SAP sends the PR for release before purchasing can proceed."
        )

    if "purchase requisition" in q or "pr" in q:
        return (
            "Example: Production requires 100 units of raw material. "
            "The user creates a PR using ME51N."
        )

    if "purchase order" in q or "po" in q:
        return (
            "Example: A company needs 100 units of RM-1001 from ABC Suppliers. "
            "The buyer creates the PO using ME21N."
        )

    if "source list" in q:
        return (
            "Example: RM-1001 can be purchased from ABC Suppliers. "
            "ME01 is used to maintain ABC Suppliers as a valid source."
        )

    if "contract" in q:
        return (
            "Example: A company agrees with a vendor to purchase "
            "10,000 units over one year under a contract."
        )

    if "scheduling agreement" in q:
        return (
            "Example: A vendor supplies 500 units every month based on "
            "scheduled delivery quantities."
        )

    if "migo" in q or "goods receipt" in q:
        return (
            "Example: 100 units arrive against PO 4500001234. "
            "The warehouse posts Goods Receipt using MIGO with movement type 101."
        )

    if "miro" in q or "invoice" in q:
        return (
            "Example: Vendor sends an invoice for goods already received. "
            "The invoice is verified and posted using MIRO."
        )

    if "obyc" in q or "account determination" in q:
        return (
            "Example: During Goods Receipt, SAP automatically determines "
            "the relevant inventory and GR/IR accounts based on configuration."
        )

    if "movement type" in q:
        return (
            "Example: Movement type 101 is commonly used for Goods Receipt "
            "against a Purchase Order."
        )

    return None


# ============================================================
# PROCESS EXTRACTION
# ============================================================

def extract_process(question):

    q = question.lower()

    if 'p2p' in q or 'procure to pay' in q:
        return (
            'PR -> PO -> GR -> MIRO'
        )


    if "purchase" in q:
        return (
            "Requirement â†’ PR â†’ RFQ â†’ Vendor Selection â†’ "
            "PO â†’ Goods Receipt â†’ Invoice Verification"
        )

    return None


# ============================================================
# TROUBLESHOOTING
# ============================================================

TROUBLESHOOTING_DATABASE = {

    "account determination": {
        "problem": "Account determination error during MM transaction.",
        "checks": [
            "Check OBYC configuration.",
            "Check valuation class.",
            "Check transaction key.",
            "Check chart of accounts.",
            "Check valuation area and relevant configuration."
        ]
    },

    "release strategy": {
        "problem": "PR or purchasing document is not releasing as expected.",
        "checks": [
            "Check release strategy configuration.",
            "Check release characteristics.",
            "Check classification values.",
            "Check release codes.",
            "Check approval status."
        ]
    },

    "purchase order": {
        "problem": "Purchase Order cannot be created or saved.",
        "checks": [
            "Check Vendor/Business Partner.",
            "Check Purchasing Organization.",
            "Check Purchasing Group.",
            "Check Company Code.",
            "Check Material and Plant.",
            "Check mandatory fields and pricing."
        ]
    },

    "goods receipt": {
        "problem": "Goods Receipt cannot be posted.",
        "checks": [
            "Check Purchase Order status.",
            "Check open PO quantity.",
            "Check Plant and Storage Location.",
            "Check movement type.",
            "Check Item OK indicator.",
            "Check stock and posting period."
        ]
    },

    "invoice": {
        "problem": "Invoice cannot be posted in MIRO.",
        "checks": [
            "Check Purchase Order.",
            "Check Goods Receipt.",
            "Check invoice quantity.",
            "Check invoice amount.",
            "Check tax information.",
            "Check GR/IR and account determination."
        ]
    }
}


def get_troubleshooting_topic(question):

    q = question.lower()

    if "account determination" in q or "obyc error" in q:
        return "account determination"

    if "release strategy" in q or "release error" in q:
        return "release strategy"

    if "purchase order error" in q or "po error" in q:
        return "purchase order"

    if "goods receipt error" in q or "migo error" in q:
        return "goods receipt"

    if "invoice error" in q or "miro error" in q:
        return "invoice"

    return None


def show_troubleshooting(topic):

    data = TROUBLESHOOTING_DATABASE.get(topic)

    if not data:
        return

    st.error(f"âš ï¸ {data['problem']}")

    st.markdown("### ðŸ”§ Troubleshooting Checks")

    for check in data["checks"]:
        st.markdown(f"- {check}")


# ============================================================
# TRANSACTION GUIDE DETECTION
# ============================================================

def get_transaction_guide_code(question):

    q = question.lower().strip()

    # Direct T-code detection
    for code in TRANSACTION_GUIDES:

        if re.search(
            rf"\b{re.escape(code.lower())}\b",
            q
        ):
            return code

    # ME51N
    me51n_phrases = [
        "create purchase requisition",
        "creating purchase requisition",
        "how to create pr",
        "how do i create pr",
        "purchase requisition steps",
        "purchase requisition procedure",
        "purchase requisition process",
        "pr creation",
        "create pr"
    ]

    if any(phrase in q for phrase in me51n_phrases):
        return "ME51N"

    # ME41
    me41_phrases = [
        "create rfq",
        "create request for quotation",
        "how to create rfq",
        "rfq steps",
        "rfq procedure",
        "rfq process"
    ]

    if any(phrase in q for phrase in me41_phrases):
        return "ME41"

    # ME21N
    me21n_phrases = [
        "create purchase order",
        "creating purchase order",
        "how to create po",
        "how do i create po",
        "purchase order steps",
        "purchase order procedure",
        "purchase order process",
        "po creation",
        "create po"
    ]

    if any(phrase in q for phrase in me21n_phrases):
        return "ME21N"

    # MIGO
    migo_phrases = [
        "goods receipt steps",
        "how to do goods receipt",
        "how to post goods receipt",
        "create goods receipt",
        "post goods receipt",
        "migo steps",
        "goods receipt procedure"
    ]

    if any(phrase in q for phrase in migo_phrases):
        return "MIGO"

    # MIRO
    miro_phrases = [
        "how to do miro",
        "invoice verification steps",
        "how to post invoice",
        "create invoice",
        "post vendor invoice",
        "miro steps",
        "invoice procedure"
    ]

    if any(phrase in q for phrase in miro_phrases):
        return "MIRO"

    # ME01
    source_list_phrases = [
        "create source list",
        "maintain source list",
        "source list steps",
        "source list procedure"
    ]

    if any(phrase in q for phrase in source_list_phrases):
        return "ME01"

    # ME11
    info_record_phrases = [
        "create info record",
        "create purchasing info record",
        "info record steps",
        "info record procedure"
    ]

    if any(phrase in q for phrase in info_record_phrases):
        return "ME11"

    # MM01
    material_phrases = [
        "create material",
        "create material master",
        "material master steps",
        "material creation",
        "how to create material"
    ]

    if any(phrase in q for phrase in material_phrases):
        return "MM01"

    # BP
    bp_phrases = [
        "create business partner",
        "create bp",
        "business partner steps",
        "business partner procedure",
        "create vendor using bp"
    ]

    if any(phrase in q for phrase in bp_phrases):
        return "BP"

    # OBYC
    obyc_phrases = [
        "account determination steps",
        "how to configure obyc",
        "obyc configuration",
        "obyc steps",
        "automatic account determination"
    ]

    if any(phrase in q for phrase in obyc_phrases):
        return "OBYC"

    return None


# ============================================================
# SHOW TRANSACTION GUIDE
# ============================================================

def show_transaction_guide(code):

    guide = TRANSACTION_GUIDES.get(code)

    if not guide:
        return

    st.markdown(
        f"## {chr(0x1F4D8)} {code} {chr(0x2014)} {guide['title']}"
    )

    st.markdown("### " + chr(0x1F3AF) + " Purpose")
    st.write(guide["purpose"])

    st.markdown("### ðŸ“ Step-by-Step Procedure")

    for index, step in enumerate(
        guide["steps"],
        start=1
    ):
        st.markdown(
            f"**Step {index}:** {step}"
        )

    st.markdown("### " + chr(0x1F4BC) + " Practical Example")

    for item in guide["example"]:
        st.markdown(f"- {item}")

    st.markdown("### 📌 Related T-Codes")

    for tcode in guide["related"]:

        description = TCODE_DATABASE.get(
            tcode,
            "SAP MM Transaction"
        )

        st.markdown(
            f"- `{tcode}` {chr(0x2014)} {description}"
        )

    st.success("Answer source: SAP MM Transaction Guide")


# ============================================================
# RAG ANSWER DISPLAY
# ============================================================

def display_rag_answer(results):

    if not results:
        st.warning(
            "No relevant information found in the SAP MM knowledge base."
        )
        return

    best = results[0]

    st.markdown("### 💡 SAP MM Answer")

    st.write(best["text"])

    st.caption(
        f"Best similarity score: {best['score']:.3f}"
    )

    if len(results) > 1:

        st.markdown("### ðŸ“š Related Information")

        for result in results[1:]:

            with st.expander(
                f"Similarity: {result['score']:.3f}"
            ):
                st.write(result["text"])


# ============================================================
# MAIN INITIALIZATION
# ============================================================

try:

    model = load_model()

    chunks = load_knowledge()

    embeddings = create_embeddings(
        model,
        chunks
    )

except Exception as e:

    st.error("âŒ Error loading SAP MM RAG system.")
    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("âš™ï¸ System Status")

    st.success("Embedding Model Loaded âœ…")

    if chunks:
        st.success(
            f"Knowledge Base Loaded: {len(chunks)} chunks âœ…"
        )
    else:
        st.warning(
            "Knowledge file is empty or missing."
        )

    st.success("RAG Search Active âœ…")

    st.success("T-Code Detection Active âœ…")

    st.success("Movement Type Detection Active âœ…")

    st.success("Troubleshooting Active âœ…")

    st.success("Transaction Guides Active âœ…")

    st.markdown("---")

    st.markdown("### 📌 Supported Transaction Guides")

    for code, guide in TRANSACTION_GUIDES.items():

        st.write(
            f"`{code}` â€” {guide['title']}"
        )


# ============================================================
# USER QUESTION
# ============================================================

st.markdown("### Ask your SAP MM Question")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: How to create a Purchase Order?"
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown("### 💡 Example Questions")

example_questions = [
    "What is SAP MM?",
    "What is Purchase Requisition?",
    "How to create a Purchase Requisition?",
    "How to create an RFQ?",
    "How to create a Purchase Order?",
    "How to do Goods Receipt?",
    "How to do Invoice Verification?",
    "How to create Source List?",
    "How to create Info Record?",
    "How to create Material Master?",
    "How to create Business Partner?",
    "What is OBYC?",
    "What is movement type 101?",
    "MIGO error",
    "MIRO error",
    "Account determination error"
]

cols = st.columns(3)

for index, example in enumerate(example_questions):

    if cols[index % 3].button(
        example,
        key=f"example_{index}"
    ):
        question = example


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:

        st.warning("Please enter a SAP MM question.")

        st.stop()

    # --------------------------------------------------------
    # PRIORITY: P2P PROCESS QUESTION
    # --------------------------------------------------------
    p2p_question = ("p2p" in question.lower() or "procure to pay" in question.lower() or "pr to po to gr to miro" in question.lower() or ("purchase requisition" in question.lower() and "purchase order" in question.lower() and "goods receipt" in question.lower() and "invoice" in question.lower()))
    if p2p_question:
        st.markdown("### ?? SAP MM Procure-to-Pay (P2P)")
        st.write("The complete purchasing flow is: PR ? PO ? GR ? MIRO")
        st.markdown("**1. PR -> ME51N**")
        st.write("Create a Purchase Requisition to request the required material or service.")
        st.markdown("**2. PO -> ME21N**")
        st.write("Create a Purchase Order and send it to the selected vendor.")
        st.markdown("**3. GR -> MIGO**")
        st.write("Receive the material against the Purchase Order. Movement type 101 is commonly used for Goods Receipt.")
        st.markdown("**4. MIRO — Invoice Verification**")
        st.write("Verify and post the vendor invoice against the purchasing documents.")
        st.markdown("### ?? Practical Example")
        st.write("Production needs 100 units of raw material ? PR is created using ME51N ? PO is created using ME21N ? goods are received using MIGO with movement type 101 ? vendor invoice is verified and posted using MIRO.")
        display_tcodes([("ME51N", "Create Purchase Requisition"), ("ME21N", "Create Purchase Order"), ("MIGO", "Goods Receipt / Goods Movement"), ("MIRO", "Invoice Verification")])
        st.stop()
    # --------------------------------------------------------
    # 1. TRANSACTION GUIDE
    # --------------------------------------------------------

    transaction_code = get_transaction_guide_code(
        question
    )

    if transaction_code:

        show_transaction_guide(
            transaction_code
        )

    else:

        # ----------------------------------------------------
        # 2. MOVEMENT TYPE
        # ----------------------------------------------------

        movement_type, movement_description = detect_movement_type(
            question
        )

        if movement_type:

            st.markdown("### 💡 SAP MM Answer")

            st.write(
                f"Movement Type **{movement_type}** â€” "
                f"{movement_description}"
            )

            if movement_type == "101":
                st.info(
                    "Movement type 101 is commonly used for "
                    "Goods Receipt against a Purchase Order."
                )

            elif movement_type == "102":
                st.info(
                    "Movement type 102 is used to reverse "
                    "a previous 101 Goods Receipt."
                )

            elif movement_type == "201":
                st.info(
                    "Movement type 201 is used for Goods Issue "
                    "to a Cost Center."
                )

            elif movement_type == "261":
                st.info(
                    "Movement type 261 is used for Goods Issue "
                    "to a Production Order."
                )

            elif movement_type == "311":
                st.info(
                    "Movement type 311 is used for transfer "
                    "between storage locations."
                )

            st.markdown("### " + chr(0x1F4BC) + " Practical Example")

            example = generate_practical_example(
                question
            )

            if example:
                st.write(example)

            display_tcodes(
                get_question_tcodes(question)
            )

        else:

            # ------------------------------------------------
            # 3. TROUBLESHOOTING
            # ------------------------------------------------

            troubleshooting_topic = get_troubleshooting_topic(
                question
            )

            if troubleshooting_topic:

                st.markdown("### ðŸ”§ SAP MM Troubleshooting")

                show_troubleshooting(
                    troubleshooting_topic
                )

                display_tcodes(
                    get_question_tcodes(question)
                )

            else:

                # --------------------------------------------
                # 4. STRUCTURED ANSWER
                # --------------------------------------------

                topic_answer = get_topic_answer(
                    question
                )

                if topic_answer:

                    st.markdown("### 💡 SAP MM Answer")

                    st.write(topic_answer)

                    example = generate_practical_example(
                        question
                    )

                    if example:

                        st.markdown(
                            "### " + chr(0x1F4BC) + " Practical Example"
                        )

                        st.write(example)

                    process = extract_process(
                        question
                    )

                    if process:

                        st.markdown(
                            "### ðŸ”„ Process"
                        )

                        st.info(process)

                    display_tcodes(
                        get_question_tcodes(question)
                    )

                else:

                    # ----------------------------------------
                    # 5. RAG SEARCH
                    # ----------------------------------------

                    with st.spinner(
                        "Searching SAP MM knowledge..."
                    ):

                        results = rag_search(
                            question,
                            model,
                            chunks,
                            embeddings,
                            top_k=3
                        )

                    display_rag_answer(
                        results
                    )

                    example = generate_practical_example(
                        question
                    )

                    if example:

                        st.markdown(
                            "### " + chr(0x1F4BC) + " Practical Example"
                        )

                        st.write(example)

                    process = extract_process(
                        question
                    )

                    if process:

                        st.markdown(
                            "### ðŸ”„ Process"
                        )

                        st.info(process)

                    display_tcodes(
                        get_question_tcodes(question)
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SAP MM RAG Assistant | "
    "Knowledge Retrieval + Transaction Guides + Troubleshooting"
)



