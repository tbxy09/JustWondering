from forge.sdk.pilot.CodeImp import CodeMonkeyRefactored
import pytest


@pytest.fixture
def code_monkey():
    return CodeMonkeyRefactored()


def test_compose_prompt(code_monkey):
    template_name = "test_template"
    instruction = "test_instruction"
    prompt = code_monkey.compose_prompt(template_name, instruction)
    assert isinstance(prompt, str)
    assert "test_instruction" in prompt


def test_implement_code_changes(code_monkey):
    instruction = "test_instruction"
    response = code_monkey.implement_code_changes(instruction)
    assert isinstance(response, list)


def test_postprocess_response(code_monkey):
    response = {"text": "test_response"}
    processed_response = code_monkey.postprocess_response(response)
    assert processed_response == "test_response"


def test_run(code_monkey):
    project_name = "test_project"
    development_plan = "test_plan"
    output = code_monkey.run(project_name, development_plan)
    assert isinstance(output, list)

# import unittest
# from unittest.mock import MagicMock, patch
# from CodeImp import CodeMonkeyRefactored


# class TestCodeMonkeyRefactored(unittest.TestCase):

#     def setUp(self):
#         self.project_name = "test_project"
#         self.development_plan = "test_plan"
#         self.code_monkey = CodeMonkeyRefactored()

#     def test_implement_code_changes(self):
#         with patch.object(self.code_monkey, 'project') as mock_project:
#             mock_monkey = MagicMock()
#             mock_monkey.implement_code_changes.return_value = "test_changes"
#             mock_project.get_files.return_value = "test_files"
#             with patch('CodeImp.CodeMonkey', return_value=mock_monkey):
#                 result = self.code_monkey.implement_code_changes(self.development_plan)
#                 mock_monkey.implement_code_changes.assert_called_once_with(None, self.development_plan)
#                 mock_project.get_files.assert_called_once()
#                 self.assertEqual(result, "test_files")

#     def test_run(self):
#         with patch.object(self.code_monkey, 'project') as mock_project:
#             mock_project.get_directory_tree.return_value = "test_directory_tree"
#             mock_project.get_files.return_value = "test_files"
#             with patch.object(self.code_monkey, 'implement_code_changes', return_value="test_changes"):
#                 result = self.code_monkey.run(self.project_name, self.development_plan)
#                 mock_project.get_directory_tree.assert_called_once_with(True)
#                 self.code_monkey.implement_code_changes.assert_called_once_with(self.development_plan)
#                 mock_project.get_files.assert_called_once()
#                 self.assertEqual(result, "test_files")
