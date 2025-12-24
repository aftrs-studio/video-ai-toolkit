"""Tests for Video AI Toolkit."""

import pytest
from pathlib import Path

from video_toolkit.config import Config
from video_toolkit.utils import (
    parse_concepts,
    sanitize_filename,
    get_output_filename,
    format_duration,
    list_models,
)


class TestParseConcepts:
    def test_single(self):
        assert parse_concepts("person") == ["person"]

    def test_multiple(self):
        assert parse_concepts("person,car,dog") == ["person", "car", "dog"]

    def test_with_spaces(self):
        assert parse_concepts("person, car, dog") == ["person", "car", "dog"]

    def test_empty(self):
        assert parse_concepts("") == []

    def test_none(self):
        assert parse_concepts(None) == []


class TestSanitizeFilename:
    def test_simple(self):
        assert sanitize_filename("person") == "person"

    def test_spaces(self):
        assert sanitize_filename("person in car") == "person_in_car"

    def test_special_chars(self):
        assert sanitize_filename("person/car:dog") == "person_car_dog"


class TestGetOutputFilename:
    def test_basic(self):
        result = get_output_filename(
            Path("/input/video.mp4"),
            "segment",
            "person",
            Path("/output"),
        )
        assert result == Path("/output/video_segment_person.mp4")

    def test_with_instance(self):
        result = get_output_filename(
            Path("/input/video.mp4"),
            "segment",
            "person",
            Path("/output"),
            instance_id=1,
        )
        assert result == Path("/output/video_segment_person_001.mp4")


class TestFormatDuration:
    def test_seconds(self):
        assert format_duration(45) == "45s"

    def test_minutes(self):
        assert format_duration(125) == "2m 5s"

    def test_hours(self):
        assert format_duration(3725) == "1h 2m 5s"


class TestListModels:
    def test_contains_categories(self):
        output = list_models()
        assert "SEGMENT" in output
        assert "DEPTH" in output
        assert "MATTING" in output
        assert "INPAINT" in output


class TestConfig:
    def test_defaults(self):
        config = Config()
        assert config.device == "cuda"
        assert config.batch_size == 8

    def test_from_env(self):
        config = Config.from_env()
        assert config is not None
