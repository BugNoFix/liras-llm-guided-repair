# Run Report

Generated at: 2026-04-04T22:17:42.900429

## Global Summary

- Total runs: 48
- Successful runs: 48
- Success rate: 100.00%
- Total iterations: 100
- Average iterations per run: 2.083
- Total prompt tokens (est.): 0
- Total response tokens (est.): 0
- Total run duration (s): 0.000

## Table By Configuration

| config_id | runs | success_count | success_rate | iterations_avg | iterations_median | iterations_max | duration_seconds_avg | prompt_tokens_total | response_tokens_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flash25Prompt4_Shot0 | 4 | 4 | 1.0 | 5.25 | 5.0 | 10 | 0.0 | 0 | 0 |
| Flash25Prompt4_Shot1 | 4 | 4 | 1.0 | 1.75 | 2.0 | 2 | 0.0 | 0 | 0 |
| Flash25Prompt4_Shot2 | 4 | 4 | 1.0 | 2.25 | 1.5 | 5 | 0.0 | 0 | 0 |
| Flash25Prompt5_Shot0 | 4 | 4 | 1.0 | 3.25 | 3.5 | 5 | 0.0 | 0 | 0 |
| Flash25Prompt5_Shot1 | 4 | 4 | 1.0 | 1.75 | 2.0 | 2 | 0.0 | 0 | 0 |
| Flash25Prompt5_Shot2 | 4 | 4 | 1.0 | 3.25 | 2.5 | 6 | 0.0 | 0 | 0 |
| Gemini31ProPrompt4_Shot0 | 4 | 4 | 1.0 | 1.5 | 1.5 | 2 | 0.0 | 0 | 0 |
| Gemini31ProPrompt4_Shot1 | 4 | 4 | 1.0 | 1.25 | 1.0 | 2 | 0.0 | 0 | 0 |
| Gemini31ProPrompt4_Shot2 | 4 | 4 | 1.0 | 1.0 | 1.0 | 1 | 0.0 | 0 | 0 |
| Gemini31ProPrompt5_Shot0 | 4 | 4 | 1.0 | 1.25 | 1.0 | 2 | 0.0 | 0 | 0 |
| Gemini31ProPrompt5_Shot1 | 4 | 4 | 1.0 | 1.0 | 1.0 | 1 | 0.0 | 0 | 0 |
| Gemini31ProPrompt5_Shot2 | 4 | 4 | 1.0 | 1.5 | 1.5 | 2 | 0.0 | 0 | 0 |

## Table By Scenario

| scenario | runs | success_count | success_rate | iterations_avg | iterations_median | iterations_max | duration_seconds_avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Scenario_011.txt | 12 | 12 | 1.0 | 2.833 | 2.0 | 10 | 0.0 |
| Scenario_016.txt | 12 | 12 | 1.0 | 1.583 | 1.0 | 5 | 0.0 |
| Scenario_029.txt | 12 | 12 | 1.0 | 2.5 | 1.5 | 9 | 0.0 |
| Scenario_06.txt | 12 | 12 | 1.0 | 1.417 | 1.0 | 3 | 0.0 |

## Table By Model Prompt Shot

| generation_model | system_prompt | shots | runs | success_count | success_rate | iterations_avg | iterations_median | iterations_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-2.5-flash | Generative/NewSp4.txt | 0 | 4 | 4 | 1.0 | 5.25 | 5.0 | 10 |
| gemini-2.5-flash | Generative/NewSp4.txt | 1 | 4 | 4 | 1.0 | 1.75 | 2.0 | 2 |
| gemini-2.5-flash | Generative/NewSp4.txt | 2 | 4 | 4 | 1.0 | 2.25 | 1.5 | 5 |
| gemini-2.5-flash | Generative/NewSp5.txt | 0 | 4 | 4 | 1.0 | 3.25 | 3.5 | 5 |
| gemini-2.5-flash | Generative/NewSp5.txt | 1 | 4 | 4 | 1.0 | 1.75 | 2.0 | 2 |
| gemini-2.5-flash | Generative/NewSp5.txt | 2 | 4 | 4 | 1.0 | 3.25 | 2.5 | 6 |
| gemini-3.1-pro-preview | Generative/NewSp4.txt | 0 | 4 | 4 | 1.0 | 1.5 | 1.5 | 2 |
| gemini-3.1-pro-preview | Generative/NewSp4.txt | 1 | 4 | 4 | 1.0 | 1.25 | 1.0 | 2 |
| gemini-3.1-pro-preview | Generative/NewSp4.txt | 2 | 4 | 4 | 1.0 | 1.0 | 1.0 | 1 |
| gemini-3.1-pro-preview | Generative/NewSp5.txt | 0 | 4 | 4 | 1.0 | 1.25 | 1.0 | 2 |
| gemini-3.1-pro-preview | Generative/NewSp5.txt | 1 | 4 | 4 | 1.0 | 1.0 | 1.0 | 1 |
| gemini-3.1-pro-preview | Generative/NewSp5.txt | 2 | 4 | 4 | 1.0 | 1.5 | 1.5 | 2 |

## Stateless Prompt 4/5 (Dedicated)

_No data available._

### Stateless Prompt 4/5 Run Details

_No data available._

## Run Details

| config_id | run_id | scenario | system_prompt | shots | generation_model | status | derived.iteration_count | derived.success | run_started_at | run_finished_at | run_metadata_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flash25Prompt4_Shot0 | 20260403_221537 | Scenario_011.txt | Generative/NewSp4.txt | 0 | gemini-2.5-flash | success | 10 | True | 2026-04-03T22:15:37.977428 | 2026-04-03T22:20:42.359236 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot0/Scenario_011/Generative/NewSp4/RUN_20260403_221537/run_metadata.json |
| Flash25Prompt4_Shot0 | 20260403_222043 | Scenario_016.txt | Generative/NewSp4.txt | 0 | gemini-2.5-flash | success | 1 | True | 2026-04-03T22:20:43.391954 | 2026-04-03T22:21:05.335201 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot0/Scenario_016/Generative/NewSp4/RUN_20260403_222043/run_metadata.json |
| Flash25Prompt4_Shot0 | 20260403_222106 | Scenario_029.txt | Generative/NewSp4.txt | 0 | gemini-2.5-flash | success | 9 | True | 2026-04-03T22:21:06.369308 | 2026-04-03T22:23:32.405550 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot0/Scenario_029/Generative/NewSp4/RUN_20260403_222106/run_metadata.json |
| Flash25Prompt4_Shot0 | 20260403_222333 | Scenario_06.txt | Generative/NewSp4.txt | 0 | gemini-2.5-flash | success | 1 | True | 2026-04-03T22:23:33.439596 | 2026-04-03T22:23:43.141724 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot0/Scenario_06/Generative/NewSp4/RUN_20260403_222333/run_metadata.json |
| Flash25Prompt4_Shot1 | 20260403_222343 | Scenario_011.txt | Generative/NewSp4.txt | 1 | gemini-2.5-flash | success | 2 | True | 2026-04-03T22:23:43.778976 | 2026-04-03T22:24:10.579486 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot1/Scenario_011/Generative/NewSp4/RUN_20260403_222343/run_metadata.json |
| Flash25Prompt4_Shot1 | 20260403_222411 | Scenario_016.txt | Generative/NewSp4.txt | 1 | gemini-2.5-flash | success | 1 | True | 2026-04-03T22:24:11.616830 | 2026-04-03T22:24:27.565922 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot1/Scenario_016/Generative/NewSp4/RUN_20260403_222411/run_metadata.json |
| Flash25Prompt4_Shot1 | 20260403_222428 | Scenario_029.txt | Generative/NewSp4.txt | 1 | gemini-2.5-flash | success | 2 | True | 2026-04-03T22:24:28.602016 | 2026-04-03T22:25:30.336716 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot1/Scenario_029/Generative/NewSp4/RUN_20260403_222428/run_metadata.json |
| Flash25Prompt4_Shot1 | 20260403_222531 | Scenario_06.txt | Generative/NewSp4.txt | 1 | gemini-2.5-flash | success | 2 | True | 2026-04-03T22:25:31.373103 | 2026-04-03T22:25:46.014193 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot1/Scenario_06/Generative/NewSp4/RUN_20260403_222531/run_metadata.json |
| Flash25Prompt4_Shot2 | 20260403_222546 | Scenario_011.txt | Generative/NewSp4.txt | 2 | gemini-2.5-flash | success | 2 | True | 2026-04-03T22:25:46.585948 | 2026-04-03T22:26:17.346041 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot2/Scenario_011/Generative/NewSp4/RUN_20260403_222546/run_metadata.json |
| Flash25Prompt4_Shot2 | 20260403_222618 | Scenario_016.txt | Generative/NewSp4.txt | 2 | gemini-2.5-flash | success | 1 | True | 2026-04-03T22:26:18.379474 | 2026-04-03T22:26:42.775609 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot2/Scenario_016/Generative/NewSp4/RUN_20260403_222618/run_metadata.json |
| Flash25Prompt4_Shot2 | 20260403_222643 | Scenario_029.txt | Generative/NewSp4.txt | 2 | gemini-2.5-flash | success | 5 | True | 2026-04-03T22:26:43.805615 | 2026-04-03T22:28:37.826083 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot2/Scenario_029/Generative/NewSp4/RUN_20260403_222643/run_metadata.json |
| Flash25Prompt4_Shot2 | 20260403_222838 | Scenario_06.txt | Generative/NewSp4.txt | 2 | gemini-2.5-flash | success | 1 | True | 2026-04-03T22:28:38.858207 | 2026-04-03T22:29:12.349712 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt4_Shot2/Scenario_06/Generative/NewSp4/RUN_20260403_222838/run_metadata.json |
| Flash25Prompt5_Shot0 | 20260403_222912 | Scenario_011.txt | Generative/NewSp5.txt | 0 | gemini-2.5-flash | success | 3 | True | 2026-04-03T22:29:12.949263 | 2026-04-03T22:30:03.364851 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot0/Scenario_011/Generative/NewSp5/RUN_20260403_222912/run_metadata.json |
| Flash25Prompt5_Shot0 | 20260403_223004 | Scenario_016.txt | Generative/NewSp5.txt | 0 | gemini-2.5-flash | success | 5 | True | 2026-04-03T22:30:04.393042 | 2026-04-03T22:31:10.828946 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot0/Scenario_016/Generative/NewSp5/RUN_20260403_223004/run_metadata.json |
| Flash25Prompt5_Shot0 | 20260403_223111 | Scenario_029.txt | Generative/NewSp5.txt | 0 | gemini-2.5-flash | success | 4 | True | 2026-04-03T22:31:11.860891 | 2026-04-03T22:32:17.177388 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot0/Scenario_029/Generative/NewSp5/RUN_20260403_223111/run_metadata.json |
| Flash25Prompt5_Shot0 | 20260403_223218 | Scenario_06.txt | Generative/NewSp5.txt | 0 | gemini-2.5-flash | success | 1 | True | 2026-04-03T22:32:18.208869 | 2026-04-03T22:32:28.341999 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot0/Scenario_06/Generative/NewSp5/RUN_20260403_223218/run_metadata.json |
| Flash25Prompt5_Shot1 | 20260403_223228 | Scenario_011.txt | Generative/NewSp5.txt | 1 | gemini-2.5-flash | success | 2 | True | 2026-04-03T22:32:28.910485 | 2026-04-03T22:33:22.218378 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot1/Scenario_011/Generative/NewSp5/RUN_20260403_223228/run_metadata.json |
| Flash25Prompt5_Shot1 | 20260403_223323 | Scenario_016.txt | Generative/NewSp5.txt | 1 | gemini-2.5-flash | success | 2 | True | 2026-04-03T22:33:23.256823 | 2026-04-03T22:33:51.944356 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot1/Scenario_016/Generative/NewSp5/RUN_20260403_223323/run_metadata.json |
| Flash25Prompt5_Shot1 | 20260403_223352 | Scenario_029.txt | Generative/NewSp5.txt | 1 | gemini-2.5-flash | success | 2 | True | 2026-04-03T22:33:52.981685 | 2026-04-03T22:34:37.158468 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot1/Scenario_029/Generative/NewSp5/RUN_20260403_223352/run_metadata.json |
| Flash25Prompt5_Shot1 | 20260403_223438 | Scenario_06.txt | Generative/NewSp5.txt | 1 | gemini-2.5-flash | success | 1 | True | 2026-04-03T22:34:38.189580 | 2026-04-03T22:34:45.770480 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot1/Scenario_06/Generative/NewSp5/RUN_20260403_223438/run_metadata.json |
| Flash25Prompt5_Shot2 | 20260403_223446 | Scenario_011.txt | Generative/NewSp5.txt | 2 | gemini-2.5-flash | success | 6 | True | 2026-04-03T22:34:46.340050 | 2026-04-03T22:36:47.319478 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot2/Scenario_011/Generative/NewSp5/RUN_20260403_223446/run_metadata.json |
| Flash25Prompt5_Shot2 | 20260403_223648 | Scenario_016.txt | Generative/NewSp5.txt | 2 | gemini-2.5-flash | success | 2 | True | 2026-04-03T22:36:48.350029 | 2026-04-03T22:37:27.984047 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot2/Scenario_016/Generative/NewSp5/RUN_20260403_223648/run_metadata.json |
| Flash25Prompt5_Shot2 | 20260403_223729 | Scenario_029.txt | Generative/NewSp5.txt | 2 | gemini-2.5-flash | success | 2 | True | 2026-04-03T22:37:29.018007 | 2026-04-03T22:38:21.935862 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot2/Scenario_029/Generative/NewSp5/RUN_20260403_223729/run_metadata.json |
| Flash25Prompt5_Shot2 | 20260403_223822 | Scenario_06.txt | Generative/NewSp5.txt | 2 | gemini-2.5-flash | success | 3 | True | 2026-04-03T22:38:22.973113 | 2026-04-03T22:38:56.546341 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Flash25Prompt5_Shot2/Scenario_06/Generative/NewSp5/RUN_20260403_223822/run_metadata.json |
| Gemini31ProPrompt4_Shot0 | 20260403_223857 | Scenario_011.txt | Generative/NewSp4.txt | 0 | gemini-3.1-pro-preview | success | 2 | True | 2026-04-03T22:38:57.119295 | 2026-04-03T22:40:54.315338 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot0/Scenario_011/Generative/NewSp4/RUN_20260403_223857/run_metadata.json |
| Gemini31ProPrompt4_Shot0 | 20260403_224055 | Scenario_016.txt | Generative/NewSp4.txt | 0 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:40:55.352699 | 2026-04-03T22:41:33.231712 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot0/Scenario_016/Generative/NewSp4/RUN_20260403_224055/run_metadata.json |
| Gemini31ProPrompt4_Shot0 | 20260403_224134 | Scenario_029.txt | Generative/NewSp4.txt | 0 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:41:34.262979 | 2026-04-03T22:41:56.060240 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot0/Scenario_029/Generative/NewSp4/RUN_20260403_224134/run_metadata.json |
| Gemini31ProPrompt4_Shot0 | 20260403_224157 | Scenario_06.txt | Generative/NewSp4.txt | 0 | gemini-3.1-pro-preview | success | 2 | True | 2026-04-03T22:41:57.095998 | 2026-04-03T22:43:19.009116 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot0/Scenario_06/Generative/NewSp4/RUN_20260403_224157/run_metadata.json |
| Gemini31ProPrompt4_Shot1 | 20260403_224319 | Scenario_011.txt | Generative/NewSp4.txt | 1 | gemini-3.1-pro-preview | success | 2 | True | 2026-04-03T22:43:19.582910 | 2026-04-03T22:44:56.592356 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot1/Scenario_011/Generative/NewSp4/RUN_20260403_224319/run_metadata.json |
| Gemini31ProPrompt4_Shot1 | 20260403_224457 | Scenario_016.txt | Generative/NewSp4.txt | 1 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:44:57.628830 | 2026-04-03T22:45:35.412079 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot1/Scenario_016/Generative/NewSp4/RUN_20260403_224457/run_metadata.json |
| Gemini31ProPrompt4_Shot1 | 20260403_224536 | Scenario_029.txt | Generative/NewSp4.txt | 1 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:45:36.446019 | 2026-04-03T22:46:25.157486 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot1/Scenario_029/Generative/NewSp4/RUN_20260403_224536/run_metadata.json |
| Gemini31ProPrompt4_Shot1 | 20260403_224626 | Scenario_06.txt | Generative/NewSp4.txt | 1 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:46:26.189203 | 2026-04-03T22:46:47.384105 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot1/Scenario_06/Generative/NewSp4/RUN_20260403_224626/run_metadata.json |
| Gemini31ProPrompt4_Shot2 | 20260403_224647 | Scenario_011.txt | Generative/NewSp4.txt | 2 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:46:47.961286 | 2026-04-03T22:47:26.432619 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot2/Scenario_011/Generative/NewSp4/RUN_20260403_224647/run_metadata.json |
| Gemini31ProPrompt4_Shot2 | 20260403_224727 | Scenario_016.txt | Generative/NewSp4.txt | 2 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:47:27.470682 | 2026-04-03T22:48:14.228406 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot2/Scenario_016/Generative/NewSp4/RUN_20260403_224727/run_metadata.json |
| Gemini31ProPrompt4_Shot2 | 20260403_224815 | Scenario_029.txt | Generative/NewSp4.txt | 2 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:48:15.265593 | 2026-04-03T22:49:00.577386 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot2/Scenario_029/Generative/NewSp4/RUN_20260403_224815/run_metadata.json |
| Gemini31ProPrompt4_Shot2 | 20260403_224901 | Scenario_06.txt | Generative/NewSp4.txt | 2 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:49:01.616962 | 2026-04-03T22:49:30.594883 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt4_Shot2/Scenario_06/Generative/NewSp4/RUN_20260403_224901/run_metadata.json |
| Gemini31ProPrompt5_Shot0 | 20260403_224931 | Scenario_011.txt | Generative/NewSp5.txt | 0 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:49:31.164663 | 2026-04-03T22:50:05.845053 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot0/Scenario_011/Generative/NewSp5/RUN_20260403_224931/run_metadata.json |
| Gemini31ProPrompt5_Shot0 | 20260403_225006 | Scenario_016.txt | Generative/NewSp5.txt | 0 | gemini-3.1-pro-preview | success | 2 | True | 2026-04-03T22:50:06.881553 | 2026-04-03T22:51:01.371197 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot0/Scenario_016/Generative/NewSp5/RUN_20260403_225006/run_metadata.json |
| Gemini31ProPrompt5_Shot0 | 20260403_225102 | Scenario_029.txt | Generative/NewSp5.txt | 0 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:51:02.408167 | 2026-04-03T22:51:45.944706 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot0/Scenario_029/Generative/NewSp5/RUN_20260403_225102/run_metadata.json |
| Gemini31ProPrompt5_Shot0 | 20260403_225146 | Scenario_06.txt | Generative/NewSp5.txt | 0 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:51:46.976589 | 2026-04-03T22:52:02.706357 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot0/Scenario_06/Generative/NewSp5/RUN_20260403_225146/run_metadata.json |
| Gemini31ProPrompt5_Shot1 | 20260403_225203 | Scenario_011.txt | Generative/NewSp5.txt | 1 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:52:03.276392 | 2026-04-03T22:52:48.529116 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot1/Scenario_011/Generative/NewSp5/RUN_20260403_225203/run_metadata.json |
| Gemini31ProPrompt5_Shot1 | 20260403_225249 | Scenario_016.txt | Generative/NewSp5.txt | 1 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:52:49.563493 | 2026-04-03T22:53:41.391228 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot1/Scenario_016/Generative/NewSp5/RUN_20260403_225249/run_metadata.json |
| Gemini31ProPrompt5_Shot1 | 20260403_225342 | Scenario_029.txt | Generative/NewSp5.txt | 1 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:53:42.425283 | 2026-04-03T22:54:25.833976 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot1/Scenario_029/Generative/NewSp5/RUN_20260403_225342/run_metadata.json |
| Gemini31ProPrompt5_Shot1 | 20260403_225426 | Scenario_06.txt | Generative/NewSp5.txt | 1 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:54:26.867682 | 2026-04-03T22:54:49.408062 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot1/Scenario_06/Generative/NewSp5/RUN_20260403_225426/run_metadata.json |
| Gemini31ProPrompt5_Shot2 | 20260403_225450 | Scenario_011.txt | Generative/NewSp5.txt | 2 | gemini-3.1-pro-preview | success | 2 | True | 2026-04-03T22:54:50.021287 | 2026-04-03T22:56:35.337510 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot2/Scenario_011/Generative/NewSp5/RUN_20260403_225450/run_metadata.json |
| Gemini31ProPrompt5_Shot2 | 20260403_225636 | Scenario_016.txt | Generative/NewSp5.txt | 2 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:56:36.375251 | 2026-04-03T22:57:22.414042 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot2/Scenario_016/Generative/NewSp5/RUN_20260403_225636/run_metadata.json |
| Gemini31ProPrompt5_Shot2 | 20260403_225723 | Scenario_029.txt | Generative/NewSp5.txt | 2 | gemini-3.1-pro-preview | success | 1 | True | 2026-04-03T22:57:23.448501 | 2026-04-03T22:58:15.022968 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot2/Scenario_029/Generative/NewSp5/RUN_20260403_225723/run_metadata.json |
| Gemini31ProPrompt5_Shot2 | 20260403_225816 | Scenario_06.txt | Generative/NewSp5.txt | 2 | gemini-3.1-pro-preview | success | 2 | True | 2026-04-03T22:58:16.055108 | 2026-04-03T22:58:41.779305 | /Users/marco/Documents/T2I/Tesi/liras-llm-guided-repair/Runs/Gemini31ProPrompt5_Shot2/Scenario_06/Generative/NewSp5/RUN_20260403_225816/run_metadata.json |
