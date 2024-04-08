import { ModelType } from '@/types/chat';
import { DBType } from '@/types/db';

export const MODEL_ICON_MAP: Record<ModelType, { label: string; icon: string }> = {
  proxyllm: { label: 'Proxy LLM', icon: '/models/chatgpt.png' },
  'flan-t5-base': { label: 'flan-t5-base', icon: '/models/google.png' },
  'vicuna-13b': { label: 'vicuna-13b', icon: '/models/vicuna.jpeg' },
  'vicuna-7b': { label: 'vicuna-7b', icon: '/models/vicuna.jpeg' },
  'vicuna-13b-v1.5': { label: 'vicuna-13b-v1.5', icon: '/models/vicuna.jpeg' },
  'vicuna-7b-v1.5': { label: 'vicuna-7b-v1.5', icon: '/models/vicuna.jpeg' },
  'codegen2-1b': { label: 'codegen2-1B', icon: '/models/vicuna.jpeg' },
  'codet5p-2b': { label: 'codet5p-2b', icon: '/models/vicuna.jpeg' },
  'chatglm-6b-int4': { label: 'chatglm-6b-int4', icon: '/models/chatglm.png' },
  'chatglm-6b': { label: 'chatglm-6b', icon: '/models/chatglm.png' },
  'chatglm2-6b': { label: 'chatglm2-6b', icon: '/models/chatglm.png' },
  'chatglm2-6b-int4': { label: 'chatglm2-6b-int4', icon: '/models/chatglm.png' },
  'guanaco-33b-merged': { label: 'guanaco-33b-merged', icon: '/models/huggingface.svg' },
  'falcon-40b': { label: 'falcon-40b', icon: '/models/falcon.jpeg' },
  'gorilla-7b': { label: 'gorilla-7b', icon: '/models/gorilla.png' },
  'gptj-6b': { label: 'ggml-gpt4all-j-v1.3-groovy.bin', icon: '' },
  chatgpt_proxyllm: { label: 'chatgpt_proxyllm', icon: '/models/chatgpt.png' },
  bard_proxyllm: { label: 'bard_proxyllm', icon: '/models/bard.gif' },
  claude_proxyllm: { label: 'claude_proxyllm', icon: '/models/claude.png' },
  wenxin_proxyllm: { label: 'wenxin_proxyllm', icon: '' },
  tongyi_proxyllm: { label: 'tongyi_proxyllm', icon: '/models/qwen2.png' },
  zhipu_proxyllm: { label: 'zhipu_proxyllm', icon: '/models/zhipu.png' },
  'llama-2-7b': { label: 'Llama-2-7b-chat-hf', icon: '/models/llama.jpg' },
  'llama-2-13b': { label: 'Llama-2-13b-chat-hf', icon: '/models/llama.jpg' },
  'llama-2-70b': { label: 'Llama-2-70b-chat-hf', icon: '/models/llama.jpg' },
  'baichuan-13b': { label: 'Baichuan-13B-Chat', icon: '/models/baichuan.png' },
  'baichuan-7b': { label: 'baichuan-7b', icon: '/models/baichuan.png' },
  'baichuan2-7b': { label: 'Baichuan2-7B-Chat', icon: '/models/baichuan.png' },
  'baichuan2-13b': { label: 'Baichuan2-13B-Chat', icon: '/models/baichuan.png' },
  'wizardlm-13b': { label: 'WizardLM-13B-V1.2', icon: '/models/wizardlm.png' },
  'llama-cpp': { label: 'ggml-model-q4_0.bin', icon: '/models/huggingface.svg' },
  'internlm-7b': { label: 'internlm-chat-7b-v1_1', icon: '/models/internlm.png' },
  'internlm-7b-8k': { label: 'internlm-chat-7b-8k', icon: '/models/internlm.png' },
  'solar-10.7b-instruct-v1.0': { label: 'solar-10.7b-instruct-v1.0', icon: '/models/solar_logo.png' },
};

export const VECTOR_ICON_MAP: Record<string, string> = {
  Chroma: '/models/chroma-logo.png',
};

export const dbMapper: Record<DBType, { label: string; icon: string; desc: string }> = {
  mssql: { label: 'MSSQL', icon: '/icons/mssql.png', desc: 'Powerful, scalable, secure relational database system by Microsoft.' },
  bigquery: { label: 'Biqquery', icon: '/icons/bigquery.png', desc: 'Flexible, scalable NoSQL document database for web and mobile apps.' },
  redis: { label: 'Redis', icon: '/icons/redis.png', desc: 'Fast, versatile in-memory data structure store as cache, DB, or broker.' },
  hive: { label: 'Hive', icon: '/icons/hive.png', desc: 'Hive database' },
  postgresql: {
    label: 'PostgreSQL',
    icon: '/icons/postgresql.png',
    desc: 'Powerful open-source relational database with extensibility and SQL standards.',
  }

};
