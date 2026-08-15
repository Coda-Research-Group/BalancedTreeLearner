"""Process-level constants shared across BATL runners.

Two kinds of value live here: numbers the paper fixes or defaults to, and
engineering tunables that more than one module (or a test) has to agree on.
Anything that is genuinely local to a single function stays at its use site.
"""

# --- Retrieval and evaluation defaults ---

DEFAULT_RETRIEVAL_TOP_K = 10
DEFAULT_TRAINING_NEIGHBORS_TOP_K = 100
DEFAULT_TRAINING_QUERY_FRACTION = 0.01
PAPER_BEAM_SIZE = 100
PAPER_ENSEMBLE_NUM_TREES = 4
DEFAULT_ENSEMBLE_MIN_TREE_MATCHES = 2

DEFAULT_ASSIGNMENT_TOP_R = 16


LARGE_DATASET_RANDOM_SUBSET_THRESHOLD = 10_000_000

DEFAULT_NUM_LEAVES = 80
"""Leaves visited per query in the default sweep point (``evaluation.num_leaves``)."""

# --- Model architecture (paper §3.2) ---

PAPER_BRANCHING_FACTOR = 256
"""Children per internal node, K."""

PAPER_TREE_HEIGHT = 2
"""Levels of routing decisions per query, H."""

PAPER_ENCODER_HIDDEN = 1024
"""Fixed by paper §3.2.1: encoder is always Linear(d,1024)->ReLU->Linear(1024,256)."""

PAPER_MODEL_EMBED_DIM = 256
"""Paper §3.2.2 default; change only in ablation studies (e.g. {64,128,256})."""

PAPER_NUM_DECODER_LAYERS = 1
"""Fixed by paper §3.2.2: the decoder is always one layer."""

PAPER_NUM_ATTENTION_HEADS = 8
"""Decoder attention heads. Also sets the CUDA attention batch guard in
``tree_update``, where the safe batch is ``65535 // num_heads``."""

PAPER_DECODER_FF_DIM = 1024
"""Feed-forward width inside the decoder layer."""

PAPER_DROPOUT = 0.1
PAPER_BALANCE_ALPHA = 1.0
"""Capacity slack in the balanced assignment; 1.0 is exactly-tight capacity."""

DEFAULT_DATASET_EMBEDDING_DIM = 96
"""Input dimensionality d. A dataset property, not a paper hyperparameter — the
default matches Deep1B and every config is expected to set its own."""

# --- Training defaults ---

DEFAULT_TRAIN_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5

PAPER_ALTERNATING_INTERVAL = 2
"""Fixed by paper §3.3: routing model and tree are updated alternately every 2 epochs."""

DEFAULT_NEIGHBOR_SEARCH_SUBSET = 500_000
"""Vectors sampled for the top-k neighbor mining that produces training labels."""

DEFAULT_NEIGHBOR_SEARCH_CHUNK_SIZE = 1_000_000
"""Rows per chunk in ``sequential_chunked`` neighbor mining."""

DEFAULT_CONVERGENCE_PATIENCE = 3
DEFAULT_CONVERGENCE_MIN_DELTA = 0.005

# --- Tree update (Algorithm 1) ---

ASSIGNMENT_ARGSORT_CHUNK_ROWS = 1_000_000
"""Rows per argsort chunk when ordering assignment probabilities. Bounds the
peak temporary allocation without changing the result."""

RANK_HISTOGRAM_EDGES = (0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256)
"""Lower bounds of the chosen-rank histogram buckets. Exact for the first few
ranks, then doubling — the point is to expose the tail, which is what decides
how many branches a top-R assignment must decode (SPEC_performance C2). The
aggregate counters alone cannot answer that: they lump every rank >= 2
together."""

STRAGGLER_COVERAGE_TARGET = 0.999
"""Share of vectors a reported top-R must cover, i.e. the quantile behind
``min_top_r_covering_999`` in the tree-update diagnostics."""

# --- GPU-resident rerank ---

DEFAULT_UPLOAD_CHUNK_ROWS = 2_000_000
"""Rows copied host->device per upload step. Bounds the staging buffer."""

DEFAULT_VRAM_HEADROOM_BYTES = 2 * 1024**3
"""VRAM left free for the model, beam search, and rerank working memory."""

DEFAULT_MAX_GATHER_BYTES = 2 * 1024**3
"""Cap on the gathered candidate-row tensor, which sets the query micro-batch."""

GATHER_FRACTION_OF_FREE_VRAM = 0.5
"""Share of post-upload free VRAM the gather may claim, leaving the rest for
the model, beam tensors, and allocator fragmentation, which are all live at
the same time as the gather."""

MIN_HEALTHY_MICRO_BATCH_ROWS = 8
"""Below this the resident path stops amortizing kernel launches.

The measured 3.09-4.92x speedup over numpy_cpu was taken at T=1/M=100, where
the micro-batch works out to ~22 queries. T=4 triples the candidates per query
on the same card, so the batch falls toward single digits and the advantage
goes with it."""

RERANK_GROUP_QUERIES = 1024
"""Queries per resident-rerank group, after sorting by candidate count.

Large enough to keep the gather batched, small enough that the widest query in
a group is close to the narrowest."""
