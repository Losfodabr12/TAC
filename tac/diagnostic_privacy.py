"""Fronteira de privacidade do diagnóstico: nenhuma string livre é confiável.

Textos públicos são reconhecidos por igualdade com os exercícios do código.
O restante vira identificador HMAC efêmero, não um hash reversível por dicionário.
O exportador nunca envia o segredo HMAC ou textos de conversas importadas.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import re
import secrets
from collections import Counter


FIELDS = set('''id run_id checkpoint last_checkpoint checkpoint_id dataset_hash corpus parent
teacher_checkpoint retention_source fixed_reference_checkpoint fixed_reference_baseline
replay_dataset_hash reference_checkpoint initialized_from resumed_from initial_hash final_hash
format version task tokenizer tokenizer_state model_config training_config config dimension labels
status kind started finished created step steps selected_step evaluated_step target_steps epochs epoch
learning_enabled auto_train auto_on_start save_interactions theme font_size ram_mb reserve_ram_mb
vram_mb cpu_threads device batch_size accumulation context width layers heads ff_width vocab_size
tie_embeddings rope learning_rate retention_weight train_seconds auto_interval_minutes seed
conversation_profile conversation_steps conversation_seconds conversation_curriculum max_file_mb
max_text_chars max_corpus_chars web_depth web_pages web_page_mb web_interval backup_hours backup_keep
research_auto research_pages research_chars research_seconds warmup seconds position_jitter profile
text_corpus pretrain_steps replay_interval promotion_policy architecture_adapted curriculum
mixture_counts imported_mixture imported_counts imported_reference fluency_reference evaluation_interval
evaluation_context model_expansion parent_checkpoint parent_config new_config retention_sampler
parameters baseline final validation fixed_baseline fixed_final previous_baseline previous_final
reference_validation natural_baseline natural_validation context_validation session_baseline
session_validation fluency_baseline fluency_validation text_baseline text_final text_replay
imported_evaluation informational_only splits active candidate by_skill test_results outputs sessions
history messages user role content text answer actual expected reference acceptable_answers
expected_immediate expected_recall immediate_correct recall_correct skill group episode turn turns
exact correct total examples groups targets exact_matches exact_accuracy accuracy token_accuracy loss
train_loss val_loss fixed_loss known_accuracy unknown_false_accepts predictions prediction intent
immediate_accuracy recall_accuracy episodes rule_matches rule_accuracy slot_drift_rate relevance_rules
topic_relevance slot_drift nonempty repeated passes_simple_rules method scope detail error
promotion_checks dataset_examples train_examples validation_examples test_examples baseline_accuracy
validation_accuracy test_accuracy test_correct threshold legacy_test_examples legacy_test_correct
legacy_test_accuracy legacy_baseline_accuracy precision_skips sampled_sources sampled_training_bytes
replay_training_bytes executed_source_samples sources samples tokens batches source_id correction_id
reviewed language legacy imported train test validation_context test_context validation_natural
test_natural validation_fluency test_fluency validation_imported test_imported reserved user_groups
user_examples program_examples treino_concluido loss_basica respostas_basicas desconhecimento
contexto_anterior dialogos_e_correcoes sessoes_com_respostas_proprias conversas_revisadas
split records valid_records accepted_examples rejected_records rejected_examples duplicates
unknown_example_count warnings rejected line reason source_line import_id examples_count
accuracy_before accuracy_after reward a b action betas eps lr initial_lr weight_decay amsgrad unknown
foreach maximize capturable differentiable fused decoupled_weight_decay
conversation_learning_rate conversation_stability conversation_sampler conversation_generation
conversation_transfer_practice conversation_document_dataset stability_policy sampler_mode
objective_history objectives pool cross_entropy kl authored_transfer_practice selection_policy imported_selection_loss
original_baseline original_natural_baseline stage_baseline promotion_reference
'''.split())
TEXT_FIELDS = set('messages history user content text answer actual expected reference acceptable_answers expected_immediate expected_recall relevance_rules'.split())
ID_FIELDS = set('id run_id checkpoint last_checkpoint checkpoint_id dataset_hash corpus parent teacher_checkpoint retention_source fixed_reference_checkpoint replay_dataset_hash reference_checkpoint initialized_from resumed_from initial_hash final_hash text_corpus imported_reference import_id'.split())
ENUMS = set('''user assistant cpu cuda cuda:0 auto dark light current 10m fluency legacy reviewed
language imported train validation test auto file url web approved aprovado rejeitado interrompido erro epoca
epoca epoca candidato pendente concluido treinando dialogue conversation neural reforco
TAC-v1 TAC-dialogue-v1 TAC-chat-v2 TAC-conversation-sft-v1 TAC-conversation-sft-v2
TAC-local-byte-bpe-v1 utf8-bytes-v1 TAC-imported-mix-v1 TAC-fluency-balanced-v1
mixed-ascii-v2 context-macro-v1 context-natural-v2 addition-0-3-v1
validation_context test_context validation_natural test_natural validation_fluency test_fluency
validation_imported test_imported interrompido aprovado ausente valido invalido
off anchor_v1 replacement deck_v1 greedy controlled anchor protected-imported-v1 legacy-internal
original_execution legacy_or_active_reference
'''.split())
DATE = re.compile(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?')
HEX = re.compile(r'(?:[0-9a-f]{32}|[0-9a-f]{64})')


class Privacy:
    def __init__(self):
        self.secret = secrets.token_bytes(32)
        self.public = set()
        self.skills = set()
        self.redactions = Counter()

    def alias(self, value, prefix='omitido'):
        code = hmac.new(self.secret, str(value).encode('utf-8',errors='replace'),hashlib.sha256).hexdigest()[:16]
        return f'[{prefix}:{code}]'

    def identifier(self, value):
        if value is None or isinstance(value,(int,bool)):
            return value
        return value if isinstance(value,str) and (HEX.fullmatch(value) or value=='addition-0-3-v1') else self.alias(value,'id')

    def key(self, value):
        value = str(value)
        return value if value in FIELDS or value in self.skills else self.alias(value,'campo')

    def text(self, value, private=False):
        if not private and value in self.public:
            return value
        self.redactions['textos_omitidos'] += 1
        return self.alias(value,'texto')

    def sanitize(self, value, key='', private=False):
        if value is None or isinstance(value,bool):
            return value
        if isinstance(value,(int,float)):
            if isinstance(value,float) and not math.isfinite(value):
                self.redactions['numeros_nao_finitos'] += 1
                return None
            return value
        if isinstance(value,str):
            if key in TEXT_FIELDS:
                return self.text(value,private)
            if key in ID_FIELDS:
                return self.identifier(value)
            if key in ('created','started','finished') and DATE.fullmatch(value):
                return value
            if key in ('skill','intent','prediction') and not private and value in self.skills:
                return value
            if not private and value in ENUMS:
                return value
            self.redactions['strings_livres_omitidas'] += 1
            return self.alias(value)
        if isinstance(value,list):
            # Falas salvas como [papel,texto] mantêm a estrutura, nunca o texto privado.
            if key in ('messages','history'):
                return [[r[0] if r[0] in ('user','assistant') else self.alias(r[0]),self.text(r[1],private)]
                        if isinstance(r,list) and len(r)==2 and all(isinstance(t,str) for t in r)
                        else self.sanitize(r,'content',private) for r in value]
            return [self.sanitize(v,key,private) for v in value]
        if isinstance(value,dict):
            result = {}
            dynamic = key in ('by_skill','dataset_examples','mixture_counts','splits','reserved')
            for k,v in value.items():
                if k in ('password','senha','token','api_key','secret','authorization','credentials','data_dir','origin','filename','path'):
                    self.redactions['campos_excluidos'] += 1
                    continue
                known = k in FIELDS or k in self.skills or dynamic
                if not known:
                    result[self.key(k)] = self.alias('valor:'+str(v))
                    self.redactions['campos_desconhecidos_omitidos'] += 1
                else:
                    result[self.key(k)] = self.sanitize(v,k,private or k=='imported_evaluation')
            return result
        self.redactions['tipos_desconhecidos'] += 1
        return None

    def load_public_exercises(self):
        """Somente os geradores autorais do pacote. Não lê Biblioteca nem importações."""
        from .conversation_lessons import make_sets
        from .context_lessons import make_context_sets
        from .natural_dialogues import make_natural_sets
        from .fluency_lessons import make_fluency_sets
        from .dialogue_lessons import examples,LEGACY_TEST
        from .transfer_practice import practice_rows
        for factory in (make_sets,make_context_sets,make_natural_sets,make_fluency_sets,lambda:{'train':practice_rows()}):
            for rows in factory().values():
                for row in rows:
                    self.skills.add(row['skill'])
                    self.public.add(row['answer'])
                    self.public.update(row.get('acceptable_answers',[]))
                    self.public.update(row.get('relevance_rules',[]))
                    self.public.update(t for r,t in row['messages'])
        for split in ('train','validation','test'):
            for row in examples(split):
                self.public.add(row['text']); self.skills.add(row['intent'])
        self.public.update(row['text'] for row in LEGACY_TEST)


def error_category(text):
    """Classificação literal de evidência; não tenta inferir uma causa oculta."""
    lower = str(text).casefold()
    if any(t in lower for t in ('cuda out of memory','vram','outofmemoryerror')):return 'memoria_gpu'
    if any(t in lower for t in ('memoryerror','limite de ram','pouca ram')):return 'memoria_ram'
    if any(t in lower for t in ('nan','floatingpointerror','gradientes inválidos','nonfinite')):return 'precisao_numerica'
    if any(t in lower for t in ('espaço insuficiente','disk full','no space left')):return 'espaco_em_disco'
    if any(t in lower for t in ('cancelled','learningdisabled','interrupção','limite de duração')):return 'interrupcao_ou_permissao'
    if any(t in lower for t in ('tokenizador','vocabulário','contexto do exercício','snapshot','dataset')):return 'dados_ou_contexto'
    return 'causa_nao_identificada'


def safe_reason(text, privacy):
    """Preserva apenas gramáticas de motivos que o próprio treinador emite."""
    if not isinstance(text,str) or not text:return {'estado':'nao_registrado','texto':None}
    prefix='Critérios não atingidos: '
    checks=set('treino_concluido loss_basica respostas_basicas desconhecimento contexto_anterior dialogos_e_correcoes sessoes_com_respostas_proprias conversas_revisadas'.split())
    if text.startswith(prefix) and set(text[len(prefix):].split(', ')) <= checks:
        return {'estado':'registrado','texto':text}
    number=r'\d+(?:\.\d+)?'
    failure=rf'(?:validação atual|teste fixo de referência|validação anterior): piora de {number}% \(limite 2%\)'
    if re.fullmatch(r'Atualização rejeitada: '+failure+r'(?:; '+failure+r')*',text):
        return {'estado':'registrado','texto':text}
    if re.fullmatch(r'O passo final \d+ foi rejeitado\. A época do passo \d+ passou novamente na validação e foi selecionada\.',text):
        return {'estado':'registrado','texto':text}
    return {'estado':'texto_livre_omitido_por_privacidade','texto':None,
            'categoria_literal':error_category(text),'referencia':privacy.alias(text,'motivo')}
