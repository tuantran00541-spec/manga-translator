from __future__ import annotations
import json, shutil
from pathlib import Path
from PIL import Image, ImageFilter

ROOT = Path('chapter38-ocr-review')
MANIFEST = ROOT / 'processed' / 'manifest.json'

# Curated by visual review against the raw chapter, not by OCR text alone.
curated = {
    'text_0dc9765322204d79': ('TO HELL WITH THE STATUS WINDOW', 'CÚT ĐI, CỬA SỔ TRẠNG THÁI!'),
    'text_c2bc081f9d8d4645': ('SO YOU ARE THE HEAVENLY DEMON THAT WOMAN SPOKE OF.', 'Vậy ra ngươi chính là Thiên Ma mà người phụ nữ đó đã nhắc tới.'),
    'text_df60b9aac93848f3': ('...YES.', '...Đúng.'),
    'text_2e9306d6e2974bf9': ('A PLEASURE TO MEET YOU, "HEAVENLY DEMON."', 'Rất hân hạnh được gặp ngươi, "Thiên Ma".'),
    'text_dae1cd14ff22455f': ('I AM CALLED "ASURA."', 'Ta được gọi là "Asura".'),
    'text_03e9258d1f6b4b40': ('TH-THAT IS...?!', 'K-Kẻ đó là...?!'),
    'text_511f04c9f6034dd6': ('COULD HE BE THAT STRANGE BASTARD FROM BEFORE?!', 'Chẳng lẽ hắn là tên quái dị lúc trước?!'),
    'text_9e778b10f9954aa1': ('ACCORDING TO IRI, THAT STRANGE BASTARD HAD BEEN IN HIDING FOR SOME TIME.', 'Theo lời Iri, tên quái dị đó đã ẩn mình suốt một thời gian.'),
    'text_53c4552aae7742f3': ('EVEN HIS WHEREABOUTS WERE UNKNOWN...', 'Ngay cả tung tích của hắn cũng không ai biết...'),
    'text_de13e216139741bb': ('THIS IS A DISASTER...', 'Thế này thì nguy rồi...'),
    'text_4833d769dc0b4ff9': ('WHY NOW, OF ALL TIMES...?!', 'Sao lại đúng lúc này chứ...?!'),
    'text_ee554178184449db': ('WHY DID HE HAVE TO APPEAR AT THIS CRITICAL MOMENT?', 'Tại sao hắn lại xuất hiện vào đúng thời khắc then chốt này?!'),
    'text_5bb7b02b32b04244': ('IF WE WASTE ANY MORE TIME,', 'Nếu còn lãng phí thêm thời gian,'),
    'text_bbec5a6b4b25447f': ("THE HEAVENLY DEMON HEART CORE WILL COMPLETELY CONSUME THE VESSEL'S BODY.", 'Thiên Ma Tâm Hạch sẽ hoàn toàn nuốt chửng cơ thể của vật chứa.'),
    'text_c92ae5be799e41d1': ("AND ONCE THAT HAPPENS, OUR CHANCE TO OBTAIN THE HEAVENLY DEMON'S SOULLESS BODY...", 'Một khi chuyện đó xảy ra, cơ hội chiếm được thân xác vô hồn của Thiên Ma...'),
    'text_51f8a2ccd97e4b24': ('WILL VANISH AS WELL...!', '...cũng sẽ tan biến...!'),
    'text_273096c3d1714f16': ('BEFORE THAT HAPPENS...', 'Trước khi chuyện đó xảy ra...'),
    'text_2968000886f44c31': ('I MUST COMPLETE THE RITUAL AT ONCE!', 'Ta phải hoàn tất nghi thức ngay lập tức!'),
    'text_68acdb1dcfe7436d': ('...HMPH. IT IS JUST AS I HEARD FROM MYO-WOL.', '...Hừm. Quả đúng như lời Myo-Wol nói.'),
    'text_3125814fadad4e65': ("YOU INTEND TO KILL THAT WOMAN, SHATTER THE HEAVENLY DEMON'S SOUL,", 'Ngươi định giết người phụ nữ đó, nghiền nát linh hồn của Thiên Ma,'),
    'text_afa5c6dd23bf44e1': ('AND CLAIM THE BODY ONCE IT BECOMES AN EMPTY SHELL.', 'rồi chiếm lấy thân xác khi nó chỉ còn là một cái vỏ rỗng.'),
    'text_a9bbe5a485a94ea3': ('BUT THINGS WILL NOT GO THE WAY YOU WANT.', 'Nhưng mọi chuyện sẽ không diễn ra như ngươi muốn đâu.'),
    'text_908f229d79c04ff3': ('I WILL TAKE THAT DREAM OF YOURS...', 'Ta sẽ cướp lấy giấc mộng đó của ngươi...'),
    'text_4694dba2fd2a4ff9': ('...AND SMASH IT TO PIECES.', '...rồi nghiền nát nó thành từng mảnh.'),
    'text_4628257ebcc8424f': ("FOCUS ON PROTECTING THAT WOMAN WITH EVERYTHING YOU'VE GOT!", 'Hãy dốc toàn lực bảo vệ người phụ nữ đó!'),
    'text_84ef885bd9594e91': ('BEFORE THE HEAVENLY DEMON AWAKENS,', 'Trước khi Thiên Ma thức tỉnh,'),
    'text_10db78482c0a42d8': ('BOY!', 'Nhóc!'),
    'text_49b52c5b96304e05': ('I WILL SUBDUE THAT RISING STAR!', 'Ta sẽ chế ngự tên tân tinh đó!'),
    'text_fef901f151164c96': ('YES! UNDERSTOOD!', 'Vâng! Rõ!'),
    'text_0b2f908888e64910': ('...DAMN IT!', '...Chết tiệt!'),
    'text_f6e02d9ea1eb4666': ('ASSASSINS, WHAT ARE YOU DOING?!', 'Sát thủ đâu! Các ngươi đang làm gì vậy?!'),
    'text_6b47dd99069246aa': ('ALL OF YOU, MOVE OUT AND STOP THEM!', 'Tất cả xuất kích! Ngăn bọn chúng lại!'),
    'text_c8f750c46fb84c31': ('KILL THAT WENCH AT ONCE...', 'Giết ả đó ngay lập tức...'),
    'text_67dffb3fe0ce4514': ('...AND COMPLETE THE RITUAL!', '...và hoàn tất nghi thức!'),
    'text_50c792ab36ec40de': ("I NEVER IMAGINED THEY'D HAVE ASSASSINS HIDDEN THROUGHOUT THE ENTIRE SPARRING ARENA...!", 'Không ngờ chúng lại cài sát thủ khắp cả võ đài tỷ thí...!'),
    'text_1789ef8f50e045d8': ('DAMN IT...!', 'Chết tiệt...!'),
    'text_f1fe32a073bf4723': ("THERE'S NO WAY I CAN HANDLE", 'Mình không thể nào đối phó nổi'),
    'text_ec3f4a9cb28c47b6': ('THIS MANY', 'nhiều tên thế này'),
    'text_d3aed6c871af47d5': ('MY OWN...', 'một mình...'),
    'text_843b0fc08c6445e2': ('...!', '...!'),
    'text_5fdfce16e2e94d11': ('MISS SO YEONHWA!', 'Tiểu thư So Yeonhwa!'),
    'text_53a5c6e5c93d42fc': ('LETTING HIM PROTECT ME WHILE I DO NOTHING.', 'Cứ để anh ấy bảo vệ mình trong khi mình chẳng làm gì...'),
    'text_f5ba93bc095a46cd': ("...I CAN'T KEEP HIDING BEHIND LORD YUDA FOREVER,", '...mình không thể cứ mãi núp sau lưng ngài Yuda,'),
    'text_a28fa63012844408': ("SO I'LL FIGHT, TOO!", 'Vậy nên tôi cũng sẽ chiến đấu!'),
    'text_a016478925a5462d': ('TOGETHER WITH LORD YUDA!', 'Cùng với ngài Yuda!'),
    'text_ee2be26b989e4e0a': ('...UNDERSTOOD.', '...Rõ.'),
    'text_e60b3b9d6180432c': ("BUT YOU ABSOLUTELY MUSTN'T OVERDO IT!", 'Nhưng cô tuyệt đối không được quá sức!'),
    'text_b12db4d705ae4752': ('!!', '!!'),
    'text_f5e8d4137c3d4e4f': ('THE COWARD WHO TUCKED HIS TAIL AND RAN AWAY...', 'Tên hèn nhát cụp đuôi bỏ chạy...'),
    'text_25903d3ab7c54610': ('...HAS COME CRAWLING BACK TO DIE!', '...lại bò về đây để chết!'),
    'text_a0bc2abd49af406c': ("DON'T WORRY. I'LL SEND YOU TO JOIN THEM SOON ENOUGH.", 'Đừng lo. Ta sẽ sớm tiễn ngươi xuống đoàn tụ với chúng.'),
    'text_551e2f1465254f99': ('YOU MUST HAVE MISSED JANGDU AND YULGAE TERRIBLY.', 'Chắc ngươi nhớ Jangdu và Yulgae lắm nhỉ.'),
    'text_832b690a58a546c4': ('SHUT YOUR MOUTH!', 'Câm miệng!'),
    'text_ace2b5d5b0c8447a': ('I STILL HAD A DEBT TO SETTLE WITH YOU.', 'Ta vẫn còn món nợ phải tính với ngươi.'),
    'text_2d027d8a00984a0b': ('...ACTUALLY, THIS WORKS OUT PERFECTLY.', '...Thật ra, thế này lại vừa hay.'),
    'text_a26778567d4a4ab3': ('THIS ROTTEN FATE BETWEEN US...', 'Cái nghiệt duyên thối nát giữa chúng ta...'),
    'text_4652a6642f074b2f': ('...ENDS HERE!!', '...kết thúc tại đây!!'),
    'text_15c546d4e44b425f': ('THAT IS EXACTLY WHAT I WANTED, TOO.', 'Đó cũng chính là điều ta muốn.'),
    'text_b7f50902233743ef': ('THAT STUBBORN LIFE OF YOURS...', 'Cái mạng dai như đỉa của ngươi...'),
    'text_a98a9a9adae947b0': ('ENDS HERE', 'SẼ KẾT THÚC Ở ĐÂY'),
    'text_44f243cdbb324927': ('FOR GOOD!!', 'VĨNH VIỄN!!'),
    'text_1a934a7eaec840a9': ('RED BLOOD SWORD ART — SEVENTH FORM', 'HỒNG HUYẾT KIẾM PHÁP — THỨC THỨ BẢY'),
    'text_42f22550a9394d33': ('THE LAST TIME I FOUGHT HIM,', 'Lần trước khi giao đấu với hắn,'),
    'text_b373fc15d88b4408': ('FROM USING MY TECHNIQUES FREELY.', 'không thể tự do thi triển chiêu thức.'),
    'text_974736679ac04f6a': ('THE CRAMPED SURROUNDINGS KEPT ME', 'địa hình chật hẹp đã khiến ta'),
    'text_60f009955bd84cc5': ('...BUT THIS TIME IS DIFFERENT.', '...Nhưng lần này thì khác.'),
    'text_f7744cc524c046ac': ('WITH A WIDE-OPEN SPACE LIKE THIS...', 'Với không gian rộng rãi thế này...'),
    'text_2f399ca7fa5b491a': ("...IT'S A COMPLETELY DIFFERENT STORY!!", '...thì tình hình hoàn toàn khác!!'),
    'text_cc36a5af809b4115': ('CELESTIAL JUDGMENT', 'PHÁN QUYẾT THIÊN GIỚI'),
    'text_95d489c8ad944430': ('HOLY SWORD OF ASTERION', 'THÁNH KIẾM ASTERION'),
    'text_3d6b0123e80a421f': ('I-IMPOSSIBLE...!', 'K-Không thể nào...!'),
    'text_c8a1fbb45ea74f7a': ('HOW COULD I POSSIBLY LOSE TO TRASH LIKE HIM...?!', 'Sao ta có thể thua một thứ rác rưởi như hắn được...?!'),
    'text_44ce70ba624b4844': ('I AM A SPECIAL-GRADE ASSASSIN OF RED BLOOD MOUNTAIN...!', 'Ta là sát thủ đặc cấp của Hồng Huyết Sơn...!'),
    'text_0cc0630c09b54829': ('LOOKS LIKE THE ONE WHO CAME CRAWLING BACK TO DIE...', 'Xem ra kẻ bò về đây để chết...'),
    'text_d0b8c4acdf9d4b55': ("...WASN'T ME AFTER ALL. IT WAS YOU.", '...rốt cuộc không phải ta. Mà là ngươi.'),
    'text_4f2ea99257724c69': ('WHAT IN THE WORLD IS GOING ON?!', 'Rốt cuộc chuyện quái gì đang xảy ra vậy?!'),
    'text_e92e21982faf4a0b': ("WHY ARE THERE ASSASSINS IN THE CLAN'S SPARRING ARENA?!", 'Tại sao trong võ đài tỷ thí của môn phái lại có sát thủ?!'),
    'text_033a61f1c30c4425': ("WAIT... DIDN'T THE PATRIARCH", 'Khoan... chẳng phải gia chủ'),
    'text_a428b4e7ff12451f': ('JUST', 'vừa mới'),
    'text_2d1b78193d1e4c13': ('GIVE THOSE ASSASSINS ORDERS...?', 'ra lệnh cho đám sát thủ đó sao...?'),
    'text_3894a5bc55b64afa': ('HOW IRRITATING.', 'Phiền thật.'),
    'text_8553335692694db4': ('WHAT EXACTLY DO YOU HAVE LEFT THAT YOU CAN DO?', 'Rốt cuộc ngươi còn có thể làm được gì nữa?'),
    'text_3cf4b3ae252645c4': ('NOW THEN...', 'Giờ thì...'),
    'text_11c23760ed734a03': ('QUIT YOUR BABBLING AND TAKE BACK THE HEAVENLY DEMON HEART CORE.', 'Bớt lải nhải đi, mau lấy lại Thiên Ma Tâm Hạch.'),
    'text_5bee3c9a9aed4a5a': ("IF ANYTHING, YOU'VE ONLY MADE THINGS EASIER FOR ME.", 'Ngược lại, ngươi chỉ khiến mọi chuyện dễ dàng hơn cho ta thôi.'),
    'text_397466ca06eb4729': ('...KGH.', '...Khự.'),
    'text_56931c5eff474721': ('I NO LONGER NEED TO SUPPRESS HIS CONSCIOUSNESS MYSELF.', 'Ta không còn cần phải tự tay áp chế ý thức của hắn nữa.'),
    'text_ceb326b4473942f5': ('SINCE YOU COMPLETELY SUBDUED THE VESSEL IN MY PLACE...', 'Vì ngươi đã hoàn toàn chế ngự vật chứa thay ta...'),
    'text_a242a47493034a10': ('...WHAT?', '...Cái gì?'),
    'text_43e746e529c84d28': ('...NOW, I NO LONGER NEED', '...Giờ thì ta không còn cần'),
    'text_b09e53d0791e4923': ('THAT BOTHERSOME DUEL TO THE DEATH.', 'trận quyết tử phiền phức đó nữa.'),
    'text_4441fba98d5b4833': ('I WILL FORCIBLY AWAKEN THE HEAVENLY DEMON HEART CORE...', 'Ta sẽ cưỡng ép đánh thức Thiên Ma Tâm Hạch...'),
    'text_8362662675764d49': ('...AND MAKE ITS POWER MINE!', '...và biến sức mạnh của nó thành của ta!'),
    'text_289163d647e44241': ('...YOU TRULY ARE A NUISANCE TO THE VERY END.', '...Ngươi đúng là phiền phức đến tận phút cuối.'),
    'text_bb439bfa0777414f': ('AS EXPECTED, I SHOULD DEAL WITH YOU FIRST...', 'Quả nhiên, ta nên xử lý ngươi trước...'),
    'text_0923b44a68a84624': ('...!', '...!'),
    'text_fb0364b8aa224d56': ('...WHAT IS THIS NOW?', '...Giờ lại là cái gì nữa đây?'),
    'text_68b500cfeccc46ef': ('WH-WHAT JUST HAPPENED?', 'V-Vừa xảy ra chuyện gì vậy?'),
    'text_bc687ff152b344ea': ('...!', '...!'),
    'text_a132d11ff9404ebb': ('HAS THE HEAVENLY DEMON REALLY AWAKENED...?', 'Chẳng lẽ Thiên Ma thật sự đã thức tỉnh...?'),
    'text_b87c1c057ffb4cff': ('NO WAY...', 'Không thể nào...'),
    'text_fb9e1953e3074244': ('AT LAST, IT IS COMPLETE...!', 'Cuối cùng... cũng hoàn thành rồi...!'),
    'text_6c5df60284704920': ("IT'S DONE...!", 'Xong rồi...!'),
    'text_fa892bdd5cd04ea3': ("AT LONG LAST, THE HEAVENLY DEMON'S POWER IS IN MY HANDS!!", 'Cuối cùng... sức mạnh của Thiên Ma đã nằm trong tay ta!!'),
    'text_db3168901bc547b6': ("AND THE HEAVENLY DEMON'S POWER DWELLING WITHIN IT...", 'Và sức mạnh của Thiên Ma đang trú ngụ bên trong nó...'),
    'text_4054c913227b4dc5': ('THAT YOUNG BODY...', 'Thân thể trẻ trung đó...'),
    'text_4c248d1d917a4477': ('ALL OF IT... IS MINE!!!', 'Tất cả... đều là của ta!!!'),
    'text_4e361b34b70e4560': ("YOU'RE TEN MILLION YEARS TOO EARLY.", 'Ngươi còn sớm mười triệu năm đấy.'),
    'text_8b87fb2792f24fec': ('YOU CLAIM THAT YOU WOULD TAKE ME, THE HEAVENLY DEMON?', 'Ngươi nói sẽ chiếm lấy ta, Thiên Ma này sao?'),
    'text_acaa01b4de5543db': ("FOR A LOWLY CREATURE WHO DOESN'T EVEN KNOW HIS PLACE", 'Một sinh vật hèn mọn còn chẳng biết thân phận mình'),
    'text_acae509ad58e45ee': ('COVET MY POWER...', 'mà dám nhòm ngó sức mạnh của ta...'),
    'text_56867b0aa82b47c0': ('WHAT ABOUT YOU?', 'Còn ngươi thì sao?'),
    'text_950013742a5544ee': ('THEN...', 'Vậy thì...'),
    'text_bb7b72e65fe24fc5': ('DO YOU POSSESS THE STRENGTH TO STAND AGAINST ME?', 'Ngươi có đủ sức mạnh để đối đầu với ta không?'),
}

drop_ids = set('''
text_1a026640a3504dab text_5c45a8d5a8134c21 text_5eff8b708d154c66 text_17d65dc6bafc4cc2 text_066b6813a9a4475d
text_f9e5171298864049 text_9223520dc855426a text_361c3d6f4b0b4fde text_e30d9664fefd4897 text_2f93620505f1414a
text_8b51816ca3b34f4a text_fdbf01696e8446e9 text_cbe92fa6b4074629 text_58dff71237eb43f5 text_d20d88ef00d24cdd
text_fe03e81585d9451c text_f0262507e4d34afc text_61cc12ee6ad34527 text_2512dd5152164bfe text_d18633720666455b
text_831ce03bdb3c4088 text_0e7cce56e0864faa text_0816fd19776a4a35 text_52dc475a750b4b54 text_b4e577155476455b
text_fdf18571b3cc4eea text_9136eb4d9efd4749 text_5cd02049670b45cd text_1e7efc83d1964694 text_48293b369b674fd9
text_d5b059415f5740e7 text_bce1f1d712b04f81 text_20635a21da8e4ef0 text_da1e60b1d1e2426c text_e8e576f5773b44d0
text_45ea0ab42110439e text_ac3bd4782fb24a51 text_8c872eb8736f4cdc text_46b16e948cb44620 text_48d7af7afaf144f4
text_9c2d7ed40c5d41ee text_85a209aaf5814923 text_fa12fb429e724c64 text_92c5c55f437c4da8 text_ac8085b8592a458c
text_47ea1bb17fc34fdb text_93edfa37a6434335 text_7361903b650b404e text_98a8caf7093a4224 text_1589bc3106fc47af
text_63f89b3cd0454a0a text_8ad8472b112d4ff9 text_21b7136f97a6438e text_7ae7c7921b2a4f3a text_4cc8624086204f67
text_7c69edd1e2b648c7 text_8fc3dc9879f3418b text_ec6deae17d6549a3 text_b18a094d404c4e10 text_0373396fa67a464f
text_d9cb8750289c4438 text_4bcdf1b0e36a4304
text_9288923eb13a4d29 text_1e0f504d51c346a5 text_c9514147749d4397
'''.split())

manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
found = set()
for page in manifest.get('pages', []):
    for obj in page.get('text_objects') or []:
        oid = str(obj.get('id') or '')
        if oid in drop_ids:
            obj['source_missing'] = True
            obj['translation'] = ''
            obj['human_review'] = 'drop_duplicate_credit_sfx_or_noise'
        if oid in curated:
            source, vi = curated[oid]
            obj['ocr_text'] = source
            obj['translation'] = vi
            obj['auto_translation'] = vi
            obj['translation_source'] = 'human_curated_vi'
            obj['ocr_source'] = 'human_curated'
            obj['ocr_quality'] = 'good'
            obj.pop('source_missing', None)
            found.add(oid)

# Merge the three split OCR rows for the single "Has the Heavenly Demon..." bubble.
for obj in manifest['pages'][75].get('text_objects') or []:
    if obj.get('id') == 'text_a132d11ff9404ebb':
        obj['region'] = {'x1': 145, 'y1': 2015, 'x2': 820, 'y2': 2260}
        obj['auto_geometry'] = dict(obj['region'])

# Keep the final challenge text completely inside the owned slice core.
for obj in manifest['pages'][85].get('text_objects') or []:
    if obj.get('id') == 'text_bb7b72e65fe24fc5':
        obj['region'] = {'x1': 80, 'y1': 2145, 'x2': 850, 'y2': 2400}
        obj['auto_geometry'] = dict(obj['region'])

# The last letters of "ENDS HERE" survived auto-inpaint on a plain white speech bubble.
# This is intentionally a deterministic white cleanup, not another LaMa pass.
p42 = ROOT / 'processed' / 'clean_011_06.png'
im = Image.open(p42).convert('RGB')
px = Image.new('RGB', (180, 100), 'white')
im.paste(px, (540, 788))
im.save(p42)
# Enlarge the text object to the actual bubble interior so Vietnamese can fit.
for obj in manifest['pages'][42].get('text_objects') or []:
    if obj.get('id') == 'text_a98a9a9adae947b0':
        obj['region'] = {'x1': 145, 'y1': 700, 'x2': 760, 'y2': 880}
        obj['auto_geometry'] = dict(obj['region'])

# Restore artwork that one low-confidence false-positive mask erased on page 34.
raw34 = Image.open(ROOT / 'raw' / 'sliced' / '009_05.png').convert('RGB')
clean34_path = ROOT / 'processed' / 'clean_009_05.png'
clean34 = Image.open(clean34_path).convert('RGB')
mask34 = Image.open(ROOT / 'processed' / 'masks' / 'page_034' / 'box_6385755648594b93.3001999a03ba714d1f66e6cb4c8e59c16513e7824181e26c87b0cb9592df6c54.png').convert('L')
mask34 = mask34.point(lambda p: 255 if p > 0 else 0)
fullmask34 = Image.new('L', clean34.size, 0)
fullmask34.paste(mask34, (386, 1272))
fullmask34 = fullmask34.filter(ImageFilter.MaxFilter(5))
clean34.paste(raw34, (0, 0), fullmask34)
clean34.save(clean34_path)
raw34.close(); clean34.close(); mask34.close(); fullmask34.close()

# "BLOOD SHADOW FLASH" is a missed free-text attack name on a plain white field.
# Remove the English lettering deterministically and add one reviewed text object.
p43 = ROOT / 'processed' / 'clean_012_00.png'
im43 = Image.open(p43).convert('RGB')
im43.paste(Image.new('RGB', (820, 350), 'white'), (40, 3010))
im43.save(p43); im43.close()
manual_id = 'text_manual_blood_shadow_flash'
manual_obj = {
    'id': manual_id,
    'shape': 'rectangle',
    'region': {'x1': 70, 'y1': 3020, 'x2': 830, 'y2': 3350},
    'source_boxes': [],
    'ocr_text': 'BLOOD SHADOW FLASH',
    'translation': 'HUYẾT ẢNH THIỂM',
    'auto_translation': 'HUYẾT ẢNH THIỂM',
    'translation_source': 'human_curated_vi',
    'ocr_source': 'human_curated',
    'ocr_quality': 'good',
    'origin': 'human_review',
    'auto_generated': False,
    'style': {
        'color': '#b31313', 'font': 'default', 'fontSize': 'auto', 'bold': True,
        'strokeWidth': '2', 'strokeColor': '#000000', 'bgColor': 'transparent',
        'cornerRadius': '0', 'horizontalAlign': 'center', 'verticalAlign': 'middle'
    },
}
objs43 = manifest['pages'][43].setdefault('text_objects', [])
objs43[:] = [o for o in objs43 if o.get('id') != manual_id]
objs43.append(manual_obj)

missing = sorted(set(curated) - found)
if missing:
    raise SystemExit(f'curated ids not found: {missing}')

MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

rows = []
for pi, page in enumerate(manifest.get('pages', [])):
    for obj in page.get('text_objects') or []:
        if obj.get('source_missing'):
            continue
        source = str(obj.get('ocr_text') or '').strip()
        vi = str(obj.get('translation') or '').strip()
        if source or vi:
            rows.append({'page_index': pi, 'id': obj.get('id'), 'source_text': source, 'translation_vi': vi, 'region': obj.get('region')})
summary = {
    'checkpoint': 'translation_review',
    'chapter_id': manifest.get('chapter_id'),
    'curated_translation_count': sum(bool(r['translation_vi']) for r in rows),
    'active_text_rows': len(rows),
    'dropped_ids': len(drop_ids),
    'manual_free_text_added': 1,
    'deterministic_clean_repairs': ['page34_restore_false_positive_artwork', 'page42_remove_residual_E', 'page43_remove_BLOOD_SHADOW_FLASH'],
    'next_action': 'human_review_translation_then_render',
}
(ROOT / 'chapter38-curated-text.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
(ROOT / 'chapter38-curation-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
