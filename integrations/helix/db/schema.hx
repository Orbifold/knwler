// ─────────────────────────────────────────────────────────────────────────────
// knwler → HelixDB schema
//
// Graph topology:
//
//   Document ──HAS_CHUNK──► Chunk
//                             ▲
//   Entity ──MENTIONED_IN────┘
//     │
//     ├──IN_COMMUNITY──► Community
//     │
//     └──RELATES_TO──────► Entity
// ─────────────────────────────────────────────────────────────────────────────

// ── Nodes ────────────────────────────────────────────────────────────────────

N::Document {
    INDEX title: String,
    summary: String,
    url: String,
    language: String,
}

// Consolidated, deduplicated entity from graph.entities
N::Entity {
    INDEX name: String,
    entity_type: String,
    description: String,
}

// Raw text chunk from the source document
N::Chunk {
    chunk_idx: U32,
    text: String,
    rephrase: String,
    chunk_tokens: U32,
    source_file: String,
}

// Community detected by Louvain algorithm over the entity graph
N::Community {
    community_id: U32,
    topics: String,        // JSON-encoded string[], e.g. '["Concept","Activity"]'
    description: String,
}

// ── Edges ────────────────────────────────────────────────────────────────────

// Document owns its chunks
E::HAS_CHUNK {
    From: Document,
    To: Chunk,
}

// Entity appears in a chunk (from graph.entities[].chunk_ids)
E::MENTIONED_IN {
    From: Entity,
    To: Chunk,
}

// Entity belongs to a detected community (from graph.entities[].community_id)
E::IN_COMMUNITY {
    From: Entity,
    To: Community,
}

// Consolidated knowledge-graph relation (from graph.relations[])
E::RELATES_TO {
    From: Entity,
    To: Entity,
    Properties: {
        relation_type: String,   // e.g. "correlated_with", "increases", "linked_to"
        description: String,
        strength: F64,           // 0.0 – 10.0
    }
}
