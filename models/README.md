# Models

All models use dynamic input `[B, 40, C, 36, 60]`, static terrain
`[8, 720, 1440]`, and return `[B, 12, 1, 36, 60]`.

| Model type | Description |
| --- | --- |
| `marscdodnet` | Separate dynamic ConvLSTM and terrain-FiLM streams. |
| `convlstm` | Instance-normalized ConvLSTM encoder--decoder. |
| `convlstm_s2s` | Vanilla ConvLSTM encoder--decoder. |
| `convgru` | ConvGRU encoder--decoder. |
| `predrnn` | PredRNN-style ST-LSTM with shared cross-layer memory. |
| `swinlstm` | Boundary-aware shifted-window recurrent transformer. |
| `attention_residual` | Attention-residual ConvLSTM baseline. |

Baselines pool static terrain to 36 x 60 and concatenate it with recurrent
inputs. MarsCDODNet processes static terrain separately and uses it to FiLM
condition decoder states.

Train any model through `python -m MarsCDODNet.models.training`; see the root
README for a complete command.
