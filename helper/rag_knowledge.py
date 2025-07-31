"""
RAG Knowledge Management Helper.

Handles indexing directories into .ragfile vector databases and managing knowledge files.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


class RAGKnowledgeManager:
    """Manages RAG knowledge bases and vector databases."""
    
    def __init__(self, knowledge_dir: Path = None):
        """Initialize RAG knowledge manager."""
        # Always use global knowledge directory at project root
        self.knowledge_dir = Path("knowledge")
        self.knowledge_dir.mkdir(exist_ok=True)
        
        console.print(f"[cyan]📚 Global RAG Knowledge directory: {self.knowledge_dir}[/]")
    
    def index_directory(self, source_dir: Path, ragfile_name: str, 
                       file_patterns: List[str] = None, exclude_patterns: List[str] = None) -> bool:
        """Index a directory into a .ragfile vector database."""
        source_dir = Path(source_dir)
        
        if not source_dir.exists():
            console.print(f"[red]❌ Source directory not found: {source_dir}[/]")
            return False
        
        # Default file patterns
        if file_patterns is None:
            file_patterns = ["*.py", "*.js", "*.ts", "*.md", "*.txt", "*.json", "*.yaml", "*.yml"]
        
        # Default exclude patterns
        if exclude_patterns is None:
            exclude_patterns = ["*/.git/*", "*/node_modules/*", "*/__pycache__/*", "*.pyc", "*/.venv/*"]
        
        console.print(f"[cyan]🔍 Indexing directory: {source_dir}[/]")
        console.print(f"[dim]Patterns: {', '.join(file_patterns)}[/]")
        
        # Collect files to index
        files_to_index = []
        for pattern in file_patterns:
            for file_path in source_dir.rglob(pattern):
                if file_path.is_file():
                    # Check if file should be excluded
                    should_exclude = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern) or exclude_pattern in str(file_path):
                            should_exclude = True
                            break
                    
                    if not should_exclude:
                        files_to_index.append(file_path)
        
        if not files_to_index:
            console.print(f"[yellow]⚠️ No files found to index in {source_dir}[/]")
            return False
        
        console.print(f"[green]📄 Found {len(files_to_index)} files to index[/]")
        
        # Create knowledge base
        knowledge_data = {
            "metadata": {
                "source_directory": str(source_dir),
                "created_at": self._get_timestamp(),
                "file_count": len(files_to_index),
                "ragfile_version": "1.0",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "documents": [],
            "embeddings": [],
            "file_metadata": []
        }
        
        # Process files with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
        ) as progress:
            
            task = progress.add_task("Processing files...", total=len(files_to_index))
            
            for file_path in files_to_index:
                try:
                    # Read file content
                    content = self._read_file_safely(file_path)
                    
                    if content:
                        # Create document chunks
                        chunks = self._chunk_document(content, file_path)
                        
                        for chunk_idx, chunk in enumerate(chunks):
                            # Generate embeddings (placeholder - would use actual embedding model)
                            embedding = self._generate_embedding(chunk)
                            
                            # Store document data
                            knowledge_data["documents"].append(chunk)
                            knowledge_data["embeddings"].append(embedding)
                            knowledge_data["file_metadata"].append({
                                "file_path": str(file_path.relative_to(source_dir)),
                                "chunk_index": chunk_idx,
                                "file_type": file_path.suffix,
                                "size": len(chunk)
                            })
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️ Error processing {file_path}: {e}[/]")
                    progress.update(task, advance=1)
                    continue
        
        # Save RAG file
        ragfile_path = self.knowledge_dir / f"{ragfile_name}.ragfile"
        
        try:
            with open(ragfile_path, 'wb') as f:
                pickle.dump(knowledge_data, f)
            
            console.print(f"[green]✅ Created RAG file: {ragfile_path}[/]")
            console.print(f"[dim]Documents: {len(knowledge_data['documents'])}, Size: {self._format_size(ragfile_path.stat().st_size)}[/]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error saving RAG file: {e}[/]")
            return False
    
    def load_ragfile(self, ragfile_name: str) -> Optional[Dict[str, Any]]:
        """Load a .ragfile vector database."""
        if not ragfile_name.endswith('.ragfile'):
            ragfile_name += '.ragfile'
        
        ragfile_path = self.knowledge_dir / ragfile_name
        
        if not ragfile_path.exists():
            console.print(f"[red]❌ RAG file not found: {ragfile_path}[/]")
            return None
        
        try:
            with open(ragfile_path, 'rb') as f:
                knowledge_data = pickle.load(f)
            
            console.print(f"[green]📚 Loaded RAG file: {ragfile_name}[/]")
            metadata = knowledge_data.get("metadata", {})
            console.print(f"[dim]Documents: {metadata.get('file_count', 0)}, Created: {metadata.get('created_at', 'Unknown')}[/]")
            
            return knowledge_data
            
        except Exception as e:
            console.print(f"[red]❌ Error loading RAG file: {e}[/]")
            return None
    
    def list_ragfiles(self) -> List[Dict[str, Any]]:
        """List all available .ragfile files."""
        ragfiles = []
        
        for ragfile_path in self.knowledge_dir.glob("*.ragfile"):
            try:
                # Load metadata only
                with open(ragfile_path, 'rb') as f:
                    knowledge_data = pickle.load(f)
                
                metadata = knowledge_data.get("metadata", {})
                ragfiles.append({
                    "name": ragfile_path.stem,
                    "path": str(ragfile_path),
                    "size": self._format_size(ragfile_path.stat().st_size),
                    "file_count": metadata.get("file_count", 0),
                    "created_at": metadata.get("created_at", "Unknown"),
                    "source_directory": metadata.get("source_directory", "Unknown")
                })
                
            except Exception as e:
                console.print(f"[yellow]⚠️ Error reading {ragfile_path}: {e}[/]")
                continue
        
        return sorted(ragfiles, key=lambda x: x["name"])
    
    def delete_ragfile(self, ragfile_name: str) -> bool:
        """Delete a .ragfile."""
        if not ragfile_name.endswith('.ragfile'):
            ragfile_name += '.ragfile'
        
        ragfile_path = self.knowledge_dir / ragfile_name
        
        if not ragfile_path.exists():
            console.print(f"[yellow]⚠️ RAG file not found: {ragfile_name}[/]")
            return False
        
        try:
            ragfile_path.unlink()
            console.print(f"[green]✅ Deleted RAG file: {ragfile_name}[/]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Error deleting RAG file: {e}[/]")
            return False
    
    def search_knowledge(self, ragfile_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents in a RAG file."""
        knowledge_data = self.load_ragfile(ragfile_name)
        
        if not knowledge_data:
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Calculate similarity scores (placeholder - would use actual similarity calculation)
        results = []
        documents = knowledge_data.get("documents", [])
        embeddings = knowledge_data.get("embeddings", [])
        metadata = knowledge_data.get("file_metadata", [])
        
        for i, (doc, embedding, meta) in enumerate(zip(documents, embeddings, metadata)):
            # Calculate similarity (placeholder)
            similarity = self._calculate_similarity(query_embedding, embedding)
            
            results.append({
                "content": doc,
                "similarity": similarity,
                "file_path": meta.get("file_path", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "file_type": meta.get("file_type", "")
            })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def _read_file_safely(self, file_path: Path) -> Optional[str]:
        """Safely read file content."""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return file_path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue
            
            console.print(f"[yellow]⚠️ Could not decode {file_path}[/]")
            return None
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Error reading {file_path}: {e}[/]")
            return None
    
    def _chunk_document(self, content: str, file_path: Path, chunk_size: int = 1000) -> List[str]:
        """Split document into chunks for embedding."""
        # Simple chunking by size (could be improved with semantic chunking)
        chunks = []
        
        # Add file context to each chunk
        file_context = f"File: {file_path.name}\n\n"
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            chunks.append(file_context + chunk)
        
        return chunks
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (placeholder implementation)."""
        # Placeholder - would use actual embedding model like sentence-transformers
        # For now, return a dummy embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float list (very basic placeholder)
        embedding = [float(b) / 255.0 for b in hash_bytes[:16]]
        
        # Pad to 384 dimensions (typical for sentence transformers)
        while len(embedding) < 384:
            embedding.extend(embedding[:min(16, 384 - len(embedding))])
        
        return embedding[:384]
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings (placeholder)."""
        # Placeholder - would use actual cosine similarity
        import random
        return random.random()  # Random similarity for demo
    
    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human readable."""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()