## Hightlights

- [ ]  **Mode Used** : all prompt designed and tunned upon GPT-3.5
- [ ]  **Reasonning Building**: decompose complex tasks into executable steps, and leverages the Rewoo/ReAct Reasoning or Debate Mode framework to enhance understanding and task breakdown.

## Overall Flow

- `ReactChatAgent.run()` is the main entry point, called by the client to execute an instruction
- It calls `_compose_prompt()` to construct the prompt for the AI assistant
- The prompt is sent to OpenAI API using `_send_msg()` to generate a response
- `_parse_output()` processes the response to extract any tool actions
- For each extracted action, the corresponding tool's `run()` is called based on `_format_function_map()`
- The tool's `run()` calls internal methods like `implement_code_changes()` to execute that step
- `implement_code_changes()` calls out to Codex for code generation
- Generated artifacts are returned back up via the tool's `run()`
- In `ReactChatAgent`, observations are used to construct context for next prompt
- Loop continues until assistant indicates task is complete

## Call Hierarchy

- `ReactChatAgent.run()`
    - `_compose_prompt()`
        - `_send_msg()`
            - Azure GPT16k API
    - `_parse_output()`
        - `_format_function_map()`
            - `Tool.run()`
                - `implement_code_changes()`
                    - Codex
- Return back up call stack
    - `Tool.run()`
        - Returns observations
    - `ReactChatAgent`
        - Constructs context
        - Loops for next prompt

So in summary, `ReactChatAgent.run()` coordinates overall flow, prompts assistant, calls tools to execute steps, collects results for context, and loops until completion. Modular design allows extending with more tools.

## Prompt Table
|name | anchor | Description |
| --- | --- | --- |
| talks_dev_coder | justwondering/forge/prompts/talks_dev_coder.md | the prompts encourage a natural conversational flow with back-and-forth between the developer and coder.There are examples and guidance provided in the comments for how to format the responses. |
| execute_commands | prompts/execute_commands.prompt | work as observation_promp,constructs the prompt to collect observations after executing, also for debugging and prompt tunning |
| implement_task | FUNCTION_CALL | function call is designed to filter out the easy task to determine the task type and ability, such as reading or writing. No reasoning is needed for non-coding tasks, as they will directly go into a bash shell tool. However, coding tasks that require reasoning will lead to a reasoning process. |


## Prompt borrowed from [gptpilot](https://github.com/Pythagora-io/gpt-pilot)
| Call | Prompt | Schema | Functions |
|-|-|-|-| 
| [`convo.send_message('development/task/request_files_for_code_changes.prompt', step_data)`](./pilot/const/function_calls.py#L132) | [`request_files_for_code_changes.prompt`](./pilot/prompts/development/task/request_files_for_code_changes.prompt) | `step_data` | |
| [`convo.send_message('development/implement_changes.prompt', impl_data, IMPLEMENT_CHANGES)`](./pilot/const/function_calls.py#L137) | [`implement_changes.prompt`](./pilot/prompts/develetopment/implement_changes.prompt) | `impl_data` | [`IMPLEMENT_CHANGES`](./pilot/const/function_calls.py#L603) |  