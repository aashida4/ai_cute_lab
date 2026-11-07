import json,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
n=json.loads((Path(__file__).resolve().parents[1]/'notebooks/02_participant_workshop.ipynb').read_text())
import unittest
import contextlib
import io

class ParticipantSafetyTests(unittest.TestCase):
    def test_workflow_and_recovery(self):
        with contextlib.redirect_stdout(io.StringIO()):
            codes={i:''.join(c['source']) for i,c in enumerate(n['cells']) if c['cell_type']=='code'}
            for s in codes.values(): compile(s,'notebook','exec')
            tmp=tempfile.TemporaryDirectory(); root=Path(tmp.name); data=root/'data'/'workshop'; (data/'images').mkdir(parents=True)
            for i in range(70): Image.new('RGB',(5,5)).save(data/'images'/f'{i}.jpg')
            pd.DataFrame({'image_path':[f'images/{i}.jpg' for i in range(70)],'species':['cat']*70,'breed':['test']*70}).to_csv(data/'metadata.csv',index=False)
            np.save(data/'embeddings.npy',np.random.default_rng(9).normal(size=(70,8)))
            def boot():
             g={'WORKSHOP_DIR':str(data)}
             exec(codes[7],g);g['display']=lambda *a,**k:None;g['clear_output']=lambda *a,**k:None
             for i in [9,11,13,19,21]: exec(codes[i],g)
             return g
            g=boot()
            for i in range(int(g['holdout_mask'].sum())):g['answer_holdout'](i%2)
            for i in range(10):g['set_label'](i%2)
            assert g['labeled_counts']()[0]==10 and g['cute_button'].disabled
            g['set_label'](1);assert g['labeled_counts']()[0]==10
            before=g['labels'].copy();held=g['holdout_labels'].copy()
            for i in [9,11]:exec(codes[i],g)
            assert np.array_equal(before,g['labels']) and np.array_equal(held,g['holdout_labels'])
            g['compare_stage']();assert 10 in g['completed_stages']; assert len(g['evaluation_history'])==5
            for row in g['evaluation_history']:
                details = row['確認用詳細']
                assert {d['画像No.'] for d in details} == set(np.flatnonzero(g['holdout_mask']))
                matching = sum(d['自分の回答'] == d['AIの予測'] for d in details)
                assert np.isclose(matching / len(details), row['確認用一致率'])
                assert len(g['disagreement_rows'](row)) == len(details) - matching
            g['show_disagreements']()
            for _, record_id in g['disagreement_choice'].options:
                g['disagreement_choice'].value = record_id
                expected = g['disagreement_rows'](g['evaluation_history'][record_id])
                assert len(g['disagreement_view'].children) == (1+len(expected) if expected else 2)
            # Rendering must not alter any training or validation answers.
            assert np.array_equal(before,g['labels']) and np.array_equal(held,g['holdout_labels'])
            g['advance_stage']();assert g['stage_target']==20
            for i in range(10):g['set_label'](i%2)
            g['compare_stage']();assert 20 in g['completed_stages']
            g['advance_stage']()
            for i in range(20):g['set_label'](i%2)
            g['advance_stage']();assert g['stage_target']==40  # must compare first
            g['compare_stage']();assert g['completed_stages']==[10,20,40]
            h=boot();assert h['labeled_counts']()[0]==40 and len(h['evaluation_history'])==15
            h['show_disagreements']()
            assert len(h['disagreement_choice'].options) == 15
            # Old histories without per-image predictions remain readable.
            saved_history = h['evaluation_history']
            h['evaluation_history'] = [{'モデル': '旧版', '学習枚数': 10}]
            h['show_disagreements']()
            assert not h['disagreement_choice'].options
            assert '旧版' in h['disagreement_view'].children[0].value
            h['evaluation_history'] = saved_history
            original_history = list(h['evaluation_history'])
            h['advance_stage']();assert h['stage_target'] is None
            for i in range(5): h['set_label'](i%2)
            h['train_model']();assert h['labeled_counts']()[0]==45 and h['model'] is not None
            h['compare_stage']()
            assert h['evaluation_history'][:15]==original_history
            assert h['completed_stages']==[10,20,40]
            h=boot();assert h['stage_target'] is None and h['labeled_counts']()[0]==45
            exec(codes[11],h);assert not h['cute_button'].disabled
            while h['current_index'] is not None: h['set_label'](1)
            assert h['labeled_counts']()[0]==int((~h['holdout_mask']).sum())
            assert h['cute_button'].disabled and np.all(h['labels'][h['holdout_mask']]==-1)
            h['train_model']()
            h['request_reset']();assert h['labeled_counts']()[0]>40
            h['confirm_reset']();assert h['labeled_counts']()[0]==0 and h['holdout_ready']() and h['stage_target']==10
            assert not h['disagreement_choice'].options and not h['disagreement_view'].children
            h['request_new_participant']();h['confirm_reset']();assert not h['holdout_ready']()
            assert len(list(root.glob('participant_progress_backup_*.json')))==2
            # corruption must not overwrite saved answers
            f=root/'participant_progress.json';f.write_text('{broken')
            try:boot();raise AssertionError('corruption accepted')
            except ValueError:pass
            assert f.read_text()=='{broken'
            print('PASS: rerun, milestones, five models, restart restore, confirmed reset, backups, corrupt file protection')
            tmp.cleanup()
