"""
Repository Contract Tests.

Any repository implementation MUST pass these tests.
This ensures backends are interchangeable.

To add a new backend:
1. Implement the Repository interface
2. Add a test class that inherits RepositoryContractTests
3. Provide a `repo` fixture that returns your implementation
"""

import pytest
from abc import ABC
from datetime import datetime

from models import Project, ProbeResponse, SkepticFeedback, EntityVerification


class RepositoryContractTests(ABC):
    """
    Contract tests that any repository must pass.

    Subclass this and provide a `repo` fixture.
    """

    # === Project Repository ===

    def test_save_and_load_project(self, repo):
        project = Project(name="test", topic="Test topic")
        repo.projects.save(project)

        loaded = repo.projects.get("test")

        assert loaded is not None
        assert loaded.name == "test"
        assert loaded.topic == "Test topic"

    def test_load_nonexistent_returns_none(self, repo):
        loaded = repo.projects.get("does-not-exist")
        assert loaded is None

    def test_project_exists(self, repo):
        assert not repo.projects.exists("test")

        repo.projects.save(Project(name="test"))

        assert repo.projects.exists("test")

    def test_list_projects_empty(self, repo):
        projects = repo.projects.list()
        assert projects == []

    def test_list_projects(self, repo):
        repo.projects.save(Project(name="project-a"))
        repo.projects.save(Project(name="project-b"))

        projects = repo.projects.list()

        names = [p.name for p in projects]
        assert "project-a" in names
        assert "project-b" in names

    def test_delete_project(self, repo):
        repo.projects.save(Project(name="to-delete"))
        assert repo.projects.exists("to-delete")

        result = repo.projects.delete("to-delete")

        assert result is True
        assert not repo.projects.exists("to-delete")

    def test_delete_nonexistent_returns_false(self, repo):
        result = repo.projects.delete("does-not-exist")
        assert result is False

    def test_save_updates_timestamp(self, repo):
        project = Project(name="test")
        original_updated = project.updated_at

        repo.projects.save(project)
        loaded = repo.projects.get("test")

        assert loaded.updated_at >= original_updated

    def test_project_with_questions(self, repo):
        from models.corpus import Question

        project = Project(
            name="test",
            questions=[Question(question="Question 1"), Question(question="Question 2")]
        )
        repo.projects.save(project)

        loaded = repo.projects.get("test")

        assert len(loaded.questions) == 2
        assert loaded.questions[0].question == "Question 1"

    def test_project_entity_lists(self, repo):
        project = Project(
            name="test",
            hidden_entities=["Noise1", "Noise2"],
            promoted_entities=["Signal"]
        )
        repo.projects.save(project)

        loaded = repo.projects.get("test")

        assert "Noise1" in loaded.hidden_entities
        assert "Signal" in loaded.promoted_entities

    # === Corpus Repository ===

    def test_corpus_append_and_load(self, repo):
        repo.projects.save(Project(name="test"))

        response = ProbeResponse(
            question_index=0,
            run_index=0,
            model="test-model",
            question="Test?",
            response="Answer",
            entities=["Entity1"]
        )
        repo.corpus.append("test", response)

        corpus = repo.corpus.get_for_project("test")

        assert len(corpus) == 1
        assert corpus[0].question == "Test?"
        assert corpus[0].entities == ["Entity1"]

    def test_corpus_append_multiple(self, repo):
        repo.projects.save(Project(name="test"))

        for i in range(5):
            repo.corpus.append("test", ProbeResponse(
                question_index=i,
                run_index=0,
                model="model",
                question=f"Q{i}",
                response=f"R{i}"
            ))

        corpus = repo.corpus.get_for_project("test")
        assert len(corpus) == 5

    def test_corpus_count(self, repo):
        repo.projects.save(Project(name="test"))

        assert repo.corpus.count("test") == 0

        repo.corpus.append("test", ProbeResponse(
            question_index=0, run_index=0, model="m", question="q", response="r"
        ))

        assert repo.corpus.count("test") == 1

    def test_corpus_iterate(self, repo):
        repo.projects.save(Project(name="test"))

        for i in range(3):
            repo.corpus.append("test", ProbeResponse(
                question_index=i, run_index=0, model="m",
                question=f"Q{i}", response=f"R{i}"
            ))

        questions = [r.question for r in repo.corpus.iterate("test")]
        assert questions == ["Q0", "Q1", "Q2"]

    # === Skeptic Feedback ===

    def test_skeptic_feedback_save_load(self, repo):
        repo.projects.save(Project(name="test"))

        feedback = SkepticFeedback(
            weakest_link="The connection is weak",
            counter_questions=["What disproves this?"],
            confidence="LOW"
        )
        repo.projects.save_skeptic_feedback("test", feedback)

        loaded = repo.projects.get_skeptic_feedback("test")

        assert loaded is not None
        assert loaded.weakest_link == "The connection is weak"
        assert len(loaded.counter_questions) == 1

    def test_skeptic_feedback_nonexistent(self, repo):
        repo.projects.save(Project(name="test"))
        loaded = repo.projects.get_skeptic_feedback("test")
        assert loaded is None

    # === Entity Verification ===

    def test_entity_verification_save_load(self, repo):
        repo.projects.save(Project(name="test"))

        verification = EntityVerification()
        verification.mark_verified("John Doe", url="http://example.com")
        verification.mark_unverified("Mystery Person")

        repo.projects.save_entity_verification("test", verification)

        loaded = repo.projects.get_entity_verification("test")

        assert loaded is not None
        assert "John Doe" in loaded.public_entities
        assert "Mystery Person" in loaded.private_entities


class TestJsonBackendContract(RepositoryContractTests):
    """Test JSON backend passes contract."""

    @pytest.fixture
    def repo(self, projects_dir):
        """Provide JSON repository with temp directory."""
        from repositories.json_backend import JsonRepository
        return JsonRepository(base_path=projects_dir)


# Future backends:
#
# class TestSqliteBackendContract(RepositoryContractTests):
#     @pytest.fixture
#     def repo(self, temp_dir):
#         from repositories.sqlite_backend import SqliteRepository
#         return SqliteRepository(db_path=temp_dir / "test.db")
#
# class TestRestBackendContract(RepositoryContractTests):
#     @pytest.fixture
#     def repo(self, mock_server):
#         from repositories.rest_backend import RestRepository
#         return RestRepository(base_url=mock_server.url)
