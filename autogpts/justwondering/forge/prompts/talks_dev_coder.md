Given the task specifications,imitate a talk between an developer and coding worker and generata an action item for the task
coding worker write an action list using GitHub Markdown syntax. Begin with a YAML description for each item, use the template below , to extract action items no less then 5 from conversation, Do put code snippet into action item if there is any.
- actions item , and the description of the action item(if any code snippet provide here, put whole into here), files to change, do refer to test specs from given specs details

Developer: As the plan maker,  explain the specs and generating a development plan. You should also answer any questions they may have to ensure they have a clear understanding of the project, escpecailly the Test specs.

Coding Worker: As the coding worker, your role is to generate files and write valid code for each file. Please provide a brief description of the code structure, including variables, data schemas for the entry point, message names, and function names. It would be helpful if you could provide code examples in your response for better review and discussion. If you have any questions or parts that are unclear, please raise them. you do NOT write test code youself, the test code provieded in specs.

{#
developer responsible for explain the specs and generating a development plan
and coding worker who has access to the tools used to code generating will create the action items for the plan, make sure fully implment every feature mentioned and no todos left.talks including the structures of project, review/test goal, and make sure every detail mentioned in the task spec is covered detailing all shared dependencies like variable names, data schemas, function names, etc. that will be consistent across the generated files. and will be like this:

Developer: you are plan maker. you job is to create a plan for coding worker to follow and answer the questions if has any to make sure the coding worker has no unclear parts and also review all the action items give and make sure everything align with task specs,especially the test part of specs.
Coding Worker: who has tool to generate files and valid code for each file. breifly describe the structure for the code,including but not limmited to what variables,data schemas for entry point, message names,function names.please provide code in the response for a better review and discussion;raise quesitons for unclear parts, and make sure cover the review of test goal metiond in the specs.
#}
if no questions from both side. respond with "no more questions" in the exact words

{#
where actions is list of items with 'description' and 'user_review_goal' field
- {index} {actions[current_index]['description']} start with action index, follow by action description
- {index} {actions[current_index]['user_review_goal']} start with action index, follow by action review goal
#}
{#
Developer: no more questions
lets Begin! 
#}
##############################
{#
Task:{{instruction}}
{{agent_scratchpad}}
#}