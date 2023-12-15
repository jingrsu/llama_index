from llama_index.bridge.pydantic import Field, PrivateAttr

from llama_index.node_parser.interface import MetadataAwareTextSplitter
from llama_index.utils import get_tokenizer

from llama_index.schema import (
    BaseNode,
    CopilotTextNode,
    NodeRelationship,
)
from llama_index.utils import get_tqdm_iterable

import json

from typing import List, Sequence, Any, Callable

"""Sentence splitter."""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks.base import CallbackManager
from llama_index.constants import DEFAULT_CHUNK_SIZE
from llama_index.node_parser.interface import MetadataAwareTextSplitter
from llama_index.node_parser.text.utils import (
    split_by_char,
    split_by_regex,
    split_by_sentence_tokenizer,
    split_by_sep,
)
from llama_index.utils import get_tokenizer

SENTENCE_CHUNK_OVERLAP = 200
CHUNKING_REGEX = "[^,.;。？！]+[,.;。？！]?"
DEFAULT_PARAGRAPH_SEP = "\n\n\n"

class CopilotTextSplitter(MetadataAwareTextSplitter):
    
    MAX_KEY_TOKENS: int = 2048
    MAX_VALUE_TOKENS: int = 2048
    
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="The token chunk size for each chunk.",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=SENTENCE_CHUNK_OVERLAP,
        description="The token overlap of each chunk when splitting.",
        gte=0,
    )
    separator: str = Field(
        default=" ", description="Default separator for splitting into words"
    )
    paragraph_separator: str = Field(
        default=DEFAULT_PARAGRAPH_SEP, description="Separator between paragraphs."
    )
    secondary_chunking_regex: str = Field(
        default=CHUNKING_REGEX, description="Backup regex for splitting into sentences."
    )

    _chunking_tokenizer_fn: Callable[[str], List[str]] = PrivateAttr()
    _tokenizer: Callable = PrivateAttr()
    _split_fns: List[Callable] = PrivateAttr()
    _sub_sentence_split_fns: List[Callable] = PrivateAttr()
    
    def __init__(
        self,
        separator: str = " ",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        tokenizer: Optional[Callable] = None,
        paragraph_separator: str = DEFAULT_PARAGRAPH_SEP,
        chunking_tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
        secondary_chunking_regex: str = CHUNKING_REGEX,
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
    ):
        """Initialize with parameters."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )

        callback_manager = callback_manager or CallbackManager([])
        self._chunking_tokenizer_fn = (
            chunking_tokenizer_fn or split_by_sentence_tokenizer()
        )
        self._tokenizer = tokenizer or get_tokenizer()

        self._split_fns = [
            split_by_sep(paragraph_separator),
            self._chunking_tokenizer_fn,
        ]

        self._sub_sentence_split_fns = [
            split_by_regex(secondary_chunking_regex),
            split_by_sep(separator),
            split_by_char(),
        ]

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            secondary_chunking_regex=secondary_chunking_regex,
            separator=separator,
            paragraph_separator=paragraph_separator,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )
    
    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError("This is not implemented in CopilotTextSplitter")
    
    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        raise NotImplementedError("This is not implemented in CopilotTextSplitter")
    
    def _parse_nodes(
        self, documents: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[CopilotTextNode]:
        all_nodes: List[CopilotTextNode] = []
        documents_with_progress = get_tqdm_iterable(documents, show_progress, "Parsing nodes")
        
        for doc in documents_with_progress:
            try:
                data = json.loads(doc.get_content())
            except:
                raise ValueError("The document is not a valid JSON")
            
            assert len(self._tokenizer(data["key"])) <= self.MAX_KEY_TOKENS, "The key is too long"
            assert len(self._tokenizer(data["value"])) <= self.MAX_VALUE_TOKENS, "The value is too long"
            
            all_nodes.append(
                CopilotTextNode(
                    data=data,
                    key=data["key"],
                    value=data["value"],
                    embedding=doc.embedding,
                    excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=doc.excluded_llm_metadata_keys,
                    metadata_seperator=doc.metadata_seperator,
                    metadata_template=doc.metadata_template,
                    text_template=doc.text_template,
                    relationships={NodeRelationship.SOURCE: doc.as_related_node_info()},
                )
            )

        return all_nodes