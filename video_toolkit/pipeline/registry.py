"""Processor registry for pipeline system."""

from typing import Optional, Type

from video_toolkit.pipeline.base import BaseProcessor, ProcessorInfo, ALL_CATEGORIES


class ProcessorRegistry:
    """Registry for discovering and accessing video processors.

    Maintains a mapping of processor IDs to their classes for use in pipelines.
    """

    _processors: dict[str, Type[BaseProcessor]] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, processor_class: Type[BaseProcessor]) -> Type[BaseProcessor]:
        """Register a processor class.

        Args:
            processor_class: Processor class to register

        Returns:
            The registered class (for use as decorator)
        """
        processor_id = processor_class.PROCESSOR_ID
        if not processor_id:
            raise ValueError(f"Processor {processor_class} has no PROCESSOR_ID")
        cls._processors[processor_id] = processor_class
        return processor_class

    @classmethod
    def discover(cls) -> None:
        """Discover and register all available processors.

        Scans video_toolkit modules for processor adapters.
        """
        if cls._discovered:
            return

        # Import adapters module to trigger registration
        from video_toolkit.pipeline import adapters
        _ = adapters  # Ensure module is loaded

        cls._discovered = True

    @classmethod
    def get(cls, processor_id: str) -> Type[BaseProcessor]:
        """Get a processor class by ID.

        Args:
            processor_id: Processor identifier

        Returns:
            Processor class

        Raises:
            KeyError: If processor not found
        """
        cls.discover()
        if processor_id not in cls._processors:
            available = ", ".join(sorted(cls._processors.keys()))
            raise KeyError(f"Unknown processor: {processor_id}. Available: {available}")
        return cls._processors[processor_id]

    @classmethod
    def list_all(cls) -> list[ProcessorInfo]:
        """List all registered processors.

        Returns:
            List of ProcessorInfo for all processors
        """
        cls.discover()
        return [p.get_info() for p in cls._processors.values()]

    @classmethod
    def list_ids(cls) -> list[str]:
        """List all registered processor IDs.

        Returns:
            List of processor IDs
        """
        cls.discover()
        return sorted(cls._processors.keys())

    @classmethod
    def by_category(cls, category: str) -> list[ProcessorInfo]:
        """Get processors by category.

        Args:
            category: Category name (enhancement, analysis, creative, composition, generation)

        Returns:
            List of ProcessorInfo for matching processors
        """
        cls.discover()
        return [
            p.get_info()
            for p in cls._processors.values()
            if p.CATEGORY == category
        ]

    @classmethod
    def categories(cls) -> list[str]:
        """Get all available categories.

        Returns:
            List of category names
        """
        return ALL_CATEGORIES.copy()

    @classmethod
    def get_random_candidates(
        cls,
        categories: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> list[Type[BaseProcessor]]:
        """Get processor candidates for random pipeline generation.

        Args:
            categories: Only include these categories (None = all)
            exclude: Exclude these processor IDs

        Returns:
            List of processor classes
        """
        cls.discover()
        exclude = exclude or []
        candidates = []

        for processor_id, processor_class in cls._processors.items():
            if processor_id in exclude:
                continue
            if categories and processor_class.CATEGORY not in categories:
                continue
            candidates.append(processor_class)

        return candidates

    @classmethod
    def format_list(cls, category: Optional[str] = None) -> str:
        """Format processor list for display.

        Args:
            category: Filter by category (None = all)

        Returns:
            Formatted string
        """
        cls.discover()

        if category:
            processors = cls.by_category(category)
            lines = [f"Processors ({category})", "=" * 50, ""]
        else:
            processors = cls.list_all()
            lines = ["All Processors", "=" * 50, ""]

        # Group by category
        by_cat: dict[str, list[ProcessorInfo]] = {}
        for p in processors:
            by_cat.setdefault(p.category, []).append(p)

        for cat in ALL_CATEGORIES:
            if cat not in by_cat:
                continue
            lines.append(f"{cat.upper()}")
            for p in by_cat[cat]:
                lines.append(f"  {p.processor_id:<15} {p.name} - {p.description}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered processors. For testing only."""
        cls._processors.clear()
        cls._discovered = False
