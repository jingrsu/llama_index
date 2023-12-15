from typing import Any, Optional, Sequence, Type, TypeVar
from llama_index.data_structs.data_structs import IndexDict
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.ingestion import run_transformations
from llama_index.schema import BaseNode, Document
from llama_index.service_context import ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.core import BaseRetriever
from llama_index.node_parser import CopilotTextSplitter

IndexType = TypeVar("IndexType", bound="CopilotIndex")

class CopilotIndex(VectorStoreIndex):
    
    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[IndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        use_async: bool = False,
        store_nodes_override: bool = False,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            use_async=use_async,
            store_nodes_override=store_nodes_override,
            show_progress=show_progress,
            **kwargs,
        )
    
    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> IndexType:
        """Create index from documents.

        Args:
            documents (Optional[Sequence[BaseDocument]]): List of documents to
                build the index from.

        """
        storage_context = storage_context or StorageContext.from_defaults()
        service_context = service_context or ServiceContext.from_defaults(node_parser=CopilotTextSplitter())
        docstore = storage_context.docstore

        with service_context.callback_manager.as_trace("index_construction"):
            for doc in documents:
                docstore.set_document_hash(doc.get_doc_id(), doc.hash)

            nodes = run_transformations(
                documents,  # type: ignore
                service_context.transformations,
                show_progress=show_progress,
                **kwargs,
            )

            return cls(
                nodes=nodes,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=show_progress,
                **kwargs,
            )
    
    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        # NOTE: lazy import
        from llama_index.indices.vector_store.retrievers import VectorIndexRetriever

        return VectorIndexRetriever(
            self,
            node_ids=list(self.index_struct.nodes_dict.values()),
            callback_manager=self._service_context.callback_manager,
            **kwargs,
        )