// ─────────────────────────────────────────────────────────────────────────────
// knwler → HelixDB queries
// ─────────────────────────────────────────────────────────────────────────────


// ── Write: nodes ─────────────────────────────────────────────────────────────

QUERY AddDocument(title: String, summary: String, url: String, language: String) =>
    doc <- AddN<Document>({
        title: title,
        summary: summary,
        url: url,
        language: language,
    })
    RETURN doc

QUERY AddEntity(name: String, entity_type: String, description: String) =>
    entity <- AddN<Entity>({
        name: name,
        entity_type: entity_type,
        description: description,
    })
    RETURN entity

QUERY AddChunk(
    doc_id: ID,
    chunk_idx: U32,
    text: String,
    rephrase: String,
    chunk_tokens: U32,
    source_file: String
) =>
    chunk <- AddN<Chunk>({
        chunk_idx: chunk_idx,
        text: text,
        rephrase: rephrase,
        chunk_tokens: chunk_tokens,
        source_file: source_file,
    })
    AddE<HAS_CHUNK>::From(doc_id)::To(chunk)
    RETURN chunk

QUERY AddCommunity(community_id: U32, topics: String, description: String) =>
    community <- AddN<Community>({
        community_id: community_id,
        topics: topics,
        description: description,
    })
    RETURN community


// ── Write: edges ─────────────────────────────────────────────────────────────

QUERY AddRelation(
    source_id: ID,
    target_id: ID,
    relation_type: String,
    description: String,
    strength: F64
) =>
    AddE<RELATES_TO>({
        relation_type: relation_type,
        description: description,
        strength: strength,
    })::From(source_id)::To(target_id)
    RETURN NONE

QUERY LinkEntityToChunk(entity_id: ID, chunk_id: ID) =>
    AddE<MENTIONED_IN>::From(entity_id)::To(chunk_id)
    RETURN NONE

QUERY LinkEntityToCommunity(entity_id: ID, community_id: ID) =>
    AddE<IN_COMMUNITY>::From(entity_id)::To(community_id)
    RETURN NONE


// ── Read: nodes ──────────────────────────────────────────────────────────────

QUERY GetAllDocuments() =>
    docs <- N<Document>
    RETURN docs

QUERY GetDocument(doc_id: ID) =>
    doc <- N<Document>(doc_id)
    RETURN doc

QUERY GetEntity(name: String) =>
    entity <- N<Entity>({name: name})
    RETURN entity

QUERY GetAllEntities() =>
    entities <- N<Entity>
    RETURN entities

QUERY GetAllCommunities() =>
    communities <- N<Community>
    RETURN communities

QUERY GetDocumentChunks(doc_id: ID) =>
    chunks <- N<Document>(doc_id)::Out<HAS_CHUNK>
    RETURN chunks


// ── Read: graph traversals ────────────────────────────────────────────────────

// All entities an entity relates to (outgoing)
QUERY GetEntityNeighbors(entity_id: ID) =>
    neighbors <- N<Entity>(entity_id)::Out<RELATES_TO>
    RETURN neighbors

// All RELATES_TO edges with their metadata (for inspection)
QUERY GetEntityRelationEdges(entity_id: ID) =>
    edges <- N<Entity>(entity_id)::OutE<RELATES_TO>
    RETURN edges::{
        relation_type: _::{relation_type},
        description: _::{description},
        strength: _::{strength},
        target: _::ToN,
    }

// All entities that point TO this entity
QUERY GetEntityIncomingNeighbors(entity_id: ID) =>
    neighbors <- N<Entity>(entity_id)::In<RELATES_TO>
    RETURN neighbors

// All chunks an entity is mentioned in
QUERY GetEntityChunks(entity_id: ID) =>
    chunks <- N<Entity>(entity_id)::Out<MENTIONED_IN>
    RETURN chunks

// All entities in a community
QUERY GetCommunityEntities(community_id: ID) =>
    entities <- N<Community>(community_id)::In<IN_COMMUNITY>
    RETURN entities

// Full neighbourhood: entity + its direct relations + their communities
QUERY GetEntitySubgraph(entity_id: ID) =>
    entity    <- N<Entity>(entity_id)
    neighbors <- entity::Out<RELATES_TO>
    community <- entity::Out<IN_COMMUNITY>::FIRST
    RETURN entity::{
        id:ID,
        name,
        entity_type,
        description,
        community: community::{community_id, topics, description},
        neighbors: neighbors::{id:ID, name, entity_type},
    }

// Count of outgoing relations per entity (useful for hub detection)
QUERY GetEntityRelationCount(entity_id: ID) =>
    count <- N<Entity>(entity_id)::Out<RELATES_TO>::COUNT
    RETURN count

// Two-hop path: entities reachable within 2 RELATES_TO hops
QUERY GetTwoHopNeighbors(entity_id: ID) =>
    hop1 <- N<Entity>(entity_id)::Out<RELATES_TO>
    hop2 <- hop1::Out<RELATES_TO>
    RETURN hop2


// ── Delete ────────────────────────────────────────────────────────────────────

QUERY DeleteDocument(doc_id: ID) =>
    DROP N<Document>(doc_id)::Out<HAS_CHUNK>
    DROP N<Document>(doc_id)
    RETURN NONE

QUERY DeleteEntity(entity_id: ID) =>
    DROP N<Entity>(entity_id)::OutE<RELATES_TO>
    DROP N<Entity>(entity_id)::InE<RELATES_TO>
    DROP N<Entity>(entity_id)::OutE<MENTIONED_IN>
    DROP N<Entity>(entity_id)::OutE<IN_COMMUNITY>
    DROP N<Entity>(entity_id)
    RETURN NONE
