"""Dados sintéticos exclusivos dos testes; não usam resultados ou pesos do usuário."""
import json

import pytest

from tac.experiment_view import ExperimentDialog, METRICS


def fixture_report(tmp_path):
    metrics={k:.5 for k in METRICS}
    summary={'active':metrics,'aggregates':[{'arm':'control','metrics':{k:{'mean':v} for k,v in metrics.items()}}],
        'rows':[{'arm':'control','seed':42,'basic_count':'1/2','context_count':'1/2','natural_count':'1/2',
            'imported_count':'0/2','eos_rate':1.,'checks_count':'1/2','result_file':'run/result.json'}]}
    (tmp_path/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    (tmp_path/'RELATORIO.md').write_text('Relatório sintético de teste.',encoding='utf-8')
    (tmp_path/'run').mkdir()
    (tmp_path/'run/result.json').write_text(json.dumps({'protection_checks':{'loss_basica':False},
        'training_seconds':1.,'probe_seconds':2.,'peak_vram_mb':None,'exposure':{'unique_examples':3}}),encoding='utf-8')
    return summary


def test_experiment_view_reads_results_and_changes_chart_without_training(tmp_path,qtapp):
    fixture_report(tmp_path)
    dialog=ExperimentDialog(tmp_path);dialog.show();qtapp.processEvents()
    assert dialog.table.rowCount()==1 and dialog.report.isReadOnly() and dialog.details.isReadOnly()
    dialog.selector.setCurrentIndex(2);qtapp.processEvents()
    assert dialog.chart.selected_metric==dialog.selector.currentData()
    assert not dialog.chart.grab().isNull()
    dialog.show_run(0)
    assert 'Ficou menos segura' in dialog.details.toPlainText()
    assert 'a mesma versão ativa anterior' in dialog.details.toPlainText()
    dialog.close()


def test_experiment_view_rejects_paths_outside_report(tmp_path,qtapp):
    summary=fixture_report(tmp_path);summary['rows'][0]['result_file']='../private.json'
    (tmp_path/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    dialog=ExperimentDialog(tmp_path)
    with pytest.raises(ValueError,match='fora da pasta'):dialog.show_run(0)
    dialog.close()
