"""Write a Nepali situation report from the same counts as the English report."""

import datetime as dt
import re

DEVANAGARI = re.compile('[ऀ-ॿ]')
DEVANAGARI_DIGITS = str.maketrans('0123456789', '०१२३४५६७८९')

# Days per month (Baishakh..Chaitra) of BS 2070-2083, i.e. 14 Apr 2013 to 13 Apr 2027. Two published tables agree on
# these years; later years differ between tables, so later dates get no BS date.
BS_FIRST_YEAR_START = (2070, dt.date(2013, 4, 14))
BS_MONTH_DAYS = {
    2070: (31, 31, 31, 32, 31, 31, 29, 30, 30, 29, 30, 30),
    2071: (31, 31, 32, 31, 31, 31, 30, 29, 30, 29, 30, 30),
    2072: (31, 32, 31, 32, 31, 30, 30, 29, 30, 29, 30, 30),
    2073: (31, 32, 31, 32, 31, 30, 30, 30, 29, 29, 30, 31),
    2074: (31, 31, 31, 32, 31, 31, 30, 29, 30, 29, 30, 30),
    2075: (31, 31, 32, 31, 31, 31, 30, 29, 30, 29, 30, 30),
    2076: (31, 32, 31, 32, 31, 30, 30, 30, 29, 29, 30, 30),
    2077: (31, 32, 31, 32, 31, 30, 30, 30, 29, 30, 29, 31),
    2078: (31, 31, 31, 32, 31, 31, 30, 29, 30, 29, 30, 30),
    2079: (31, 31, 32, 31, 31, 31, 30, 29, 30, 29, 30, 30),
    2080: (31, 32, 31, 32, 31, 30, 30, 30, 29, 29, 30, 30),
    2081: (31, 32, 31, 32, 31, 30, 30, 30, 29, 30, 29, 31),
    2082: (31, 31, 32, 31, 31, 31, 30, 29, 30, 29, 30, 30),
    2083: (31, 31, 32, 31, 31, 31, 30, 29, 30, 29, 30, 30),
}
MONTHS_NE = ('बैशाख', 'जेठ', 'असार', 'साउन', 'भदौ', 'असोज', 'कात्तिक', 'मंसिर', 'पुस', 'माघ', 'फागुन', 'चैत')
MONTHS_EN = (
    'Baishakh',
    'Jestha',
    'Asar',
    'Shrawan',
    'Bhadra',
    'Ashwin',
    'Kartik',
    'Mangsir',
    'Poush',
    'Magh',
    'Falgun',
    'Chaitra',
)

# Pipeline data warnings (pipeline.data_check) with a Nepali text, keyed by the start of the English warning.
DATA_WARNINGS_NE = {
    'No Sentinel-1': 'यस क्षेत्र र मितिका लागि Sentinel-1 राडार जोडी (एउटै कक्षपथ, घटनाअघि र पछि) फेला परेन। Sentinel-1 को '
    'डेटा अक्टोबर २०१४ देखि मात्र उपलब्ध छ, र भर्खरको मितिका लागि घटनापछिको तस्बिर अझै आइनसकेको हुन सक्छ।',
    'No clear Sentinel-2 image after': 'घटनापछि बादलरहित Sentinel-2 तस्बिर फेला परेन (बादल वा तस्बिर नभएको): अप्टिकल प्रमाण '
    'उपलब्ध छैन।',
    'No clear Sentinel-2 image before': 'घटनाअघि बादलरहित Sentinel-2 तस्बिर फेला परेन: अप्टिकल परिवर्तन गणना गर्न सकिँदैन।',
}
TRANSLATED_WARNING_PREFIXES = ('No Sentinel-1', 'No clear Sentinel-2')

PART_LABELS_NE = dict(
    hazard='बाढी / लेदो नक्सा', buildings='भवन क्षति', roads='सडक र पुल क्षति', cutoff='सडक सम्पर्क विच्छेद भएका बस्ती'
)
ACCESS_NE = dict(
    walk_in='सडक पुगेको ठाउँबाट पैदल',
    air_only='नक्सामा सडक/गोरेटो छैन',
    detour='लामो घुमाउरो बाटो',
    road='सडक',
    no_road_pre='घटनाअघि नै सडक थिएन',
)
CONFIDENCE_NE = dict(high='बलियो प्रमाण', possible='सम्भावित', bridge='पुल भत्किएमा')
ROAD_CLASS_NE = dict(major='राजमार्ग / मुख्य सडक', secondary='सहायक सडक', local='स्थानीय सडक', track='कच्ची सडक')
PRIORITY_ROWS = 10

# "सडक सम्पर्क विच्छेद" is the NDRRMA and press term for settlements cut off by road; "सम्पर्कविहीन" means missing people.
NOT_COMPUTED_NE = 'गणना गरिएन (माथिको टिप्पणी हेर्नुहोस्)'
STRONG_EVIDENCE_RULE_NE = (
    'सडक अवरोधको *बलियो प्रमाण* = पछिल्ला तस्बिरमा पनि देखिने अप्टिकल परिवर्तन, दुई फरक कक्षपथबाट देखिएको राडार परिवर्तन, '
    'वा राडार र अप्टिकल दुवैमा परिवर्तन (AI मोडेलले उही राडार तस्बिर पढ्ने भएकाले दोस्रो स्रोत मानिँदैन)।'
)
ROAD_BREAK_RULE_NE = 'सडकको टुक्रा (≤ ५० मिटर) को ≥ ५०% वा ५० मिटर लगातार भाग प्रभावित क्षेत्रमा परे *अवरुद्ध हुने उच्च सम्भावना*।'
ROAD_ACCESS_RULE_NE = 'सडक पहुँच = ५०० मिटरभित्र मोटरबाटो'


def bs_date(day):
    """(year, month, day) in Bikram Sambat, or None outside the table."""
    offset_days = (day - BS_FIRST_YEAR_START[1]).days
    if offset_days < 0:
        return None
    for year in sorted(BS_MONTH_DAYS):
        for month, month_days in enumerate(BS_MONTH_DAYS[year], 1):
            if offset_days < month_days:
                return year, month, offset_days + 1
            offset_days -= month_days
    return None


def bs_text_ne(iso):
    bs = bs_date(dt.date.fromisoformat(iso[:10]))
    return '' if bs is None else f'{devanagari(bs[2])} {MONTHS_NE[bs[1] - 1]} {devanagari(bs[0])}'


def bs_text_en(iso):
    bs = bs_date(dt.date.fromisoformat(iso[:10]))
    return '' if bs is None else f'{bs[2]} {MONTHS_EN[bs[1] - 1]} {bs[0]} BS'


def devanagari(value):
    """Text of value with Devanagari digits."""
    return str(value).translate(DEVANAGARI_DIGITS)


def nepali_int(value):
    """Rounded integer with Nepali digit grouping in Devanagari: 1234567 -> १२,३४,५६७."""
    number = round(float(value))
    digits = str(abs(number))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ','.join([*groups, tail])
    return devanagari(('-' if number < 0 else '') + digits)


def nepali_decimal(value):
    """Number with one decimal and Nepali digit grouping: 1234.5 -> १,२३४.५."""
    value = float(value)
    integer_part, _, decimal_part = f'{abs(value):.1f}'.partition('.')
    return ('-' if value < 0 else '') + nepali_int(int(integer_part)) + devanagari('.' + decimal_part)


def nepali_km(value):
    """Kilometres rounded as in the English report: one decimal below 100, whole numbers from 100."""
    value = float(value)
    return nepali_int(value) if abs(value) >= 100 else nepali_decimal(value)


def local_name(label):
    """'Syabru Bensi (स्याप्रु बेसी)' -> 'स्याप्रु बेसी'; a name without a Devanagari form stays as it is."""
    label = str(label)
    if DEVANAGARI.search(label) and not label.endswith(')'):
        return label
    match = re.match(r'^(.*) \((.*)\)$', label)
    if match and DEVANAGARI.search(match.group(2)):
        return match.group(2)
    return label


def label_ne(label, place, near):
    """Nepali label of a settlement unit; outlying buildings are named after the nearest settlement."""
    if place == 'outlying' and near:
        return local_name(near) + ' नजिकका छरिएका भवन'
    return local_name(label)


def table_cell(text):
    """Text for a markdown table cell; an OSM name with '|' must not split the cell."""
    return ' '.join(str(text).replace('|', '/').split())


def inputs_ne(text):
    """Nepali wording of the English Sentinel-1 / Sentinel-2 input dates text; the dates stay as month-day."""
    for english, nepali in (
        ('first clear after', 'घटनापछि पहिलो बादलरहित'),
        ('pre ', 'अघि '),
        ('post ', 'पछि '),
        ('after ', 'पछि '),
        (' dates ', ' मिति '),
        ('none', 'छैन'),
    ):
        text = text.replace(english, nepali)
    return text


def date_ne(iso):
    if not iso:
        return 'प्रयोग गरिएका तस्बिर'
    day = dt.date.fromisoformat(str(iso)[:10])
    bs = bs_text_ne(iso)
    ad = f'{day.day} {day:%b %Y}'
    return f'{bs} ({ad})' if bs else ad


def walk_ne(row):
    if row.get('access') != 'walk_in' or row.get('walk_km') is None:
        return '—'
    hours = row.get('walk_h')
    duration = f' (लगभग {devanagari(f"{max(0.5, round(hours * 2) / 2):g}")} घण्टा)' if hours is not None else ''
    roadhead = local_name(row['roadhead']) + ' नजिक ' if row.get('roadhead') else ''
    return f'{roadhead}सडक पुगेको ठाउँबाट {nepali_km(row["walk_km"])} कि.मि.{duration}'


def area_ne(municipality, ward):
    municipality = None if municipality in (None, '', 'None', 'nan') else local_name(municipality)
    ward = None if ward in (None, '', 'None', 'nan') else local_name(ward)
    if municipality and ward:
        return f'{municipality}, {ward}'
    return municipality or ward


def repair_ne(repair):
    length_m = repair.get('length_m') or 0
    length = nepali_km(length_m / 1000) + ' कि.मि.' if length_m >= 1000 else nepali_int(length_m) + ' मिटर'
    road = repair.get('road') or ROAD_CLASS_NE.get(repair.get('highway'), 'सडक')
    near = f' मा {local_name(repair["near"])} नजिक' if repair.get('near') else ''
    bridge = '(पुलसहित) ' if repair.get('bridge') else ''
    unit = 'भवन' if repair.get('gain_unit', 'buildings') == 'buildings' else 'बस्ती'
    settlement_count = repair.get('n_settlements') or 0
    settlements = f' ({devanagari(settlement_count)} बस्ती)' if settlement_count else ''
    gain = nepali_int(repair.get('gain') or 0)
    return f'{road}{near} {length} {bridge}मर्मत → {gain} {unit} पुनः जोडिन्छन्{settlements}'


def header_lines(context):
    areas = context['areas']
    where = ', '.join(local_name(name) for name in (areas.get('districts') or [])[:3]) or ', '.join(
        local_name(name) for name in (areas.get('municipalities') or [])[:3]
    )
    title = f'# स्थिति प्रतिवेदन — {context["title"]}' + (f' — {where}' if where else '')
    bs_date_ne = context['bs_date_ne']
    generated = f'तयार: {context["generated"]} UTC'
    if not context['newest_date']:
        event_date = bs_date_ne or context['date']
        return [
            title,
            f'*घटनापछिको प्रयोगयोग्य भू-उपग्रह तस्बिर फेला परेन · घटना: {event_date} · {generated} · शैक्षिक प्रोटोटाइप।*',
            '',
        ]
    event_date = f'{bs_date_ne} ({context["date"]})' if bs_date_ne else context['date']
    subtitle = (
        f'*भू-उपग्रह तस्बिर **{date_ne(context["newest_date"])}** सम्मको (घटनापछिको पहिलो: '
        f'{date_ne(context["first_post_date"])}) · घटना: {event_date} · {generated} · स्वचालित भू-उपग्रह विश्लेषण, '
        'स्थलगत रूपमा प्रमाणित होइन (शैक्षिक प्रोटोटाइप)।*'
    )
    return [title, subtitle, '']


def notes(availability):
    lines = [
        f'> **गणना गरिएन:** {PART_LABELS_NE[part]} (कारण: {availability[part]})'
        for part in ('hazard', 'buildings', 'roads', 'cutoff')
        if availability[part]
    ]
    is_dl_used, dl_text = availability['dl']
    if not is_dl_used:
        lines.append(f'> **AI मोडेल प्रयोग गरिएन** (कारण: {dl_text})')
    warnings = availability['warnings']
    for warning in warnings:
        lines += [
            f'> **डेटा चेतावनी:** {text}' for prefix, text in DATA_WARNINGS_NE.items() if warning.startswith(prefix)
        ]
    lines += [
        f'> **चेतावनी (अंग्रेजीमा):** {warning}'
        for warning in warnings
        if not warning.startswith(TRANSLATED_WARNING_PREFIXES)
    ]
    return lines


def units_ne(unit_count):
    return f' ({nepali_int(unit_count)} बस्तीमा)' if unit_count else ''


def access_rows(access):
    if access is None:
        return [('सडक पहुँच नभएका', NOT_COMPUTED_NE, '')]
    if not access['has_buildings']:
        any_count = access['settlements_high'] + access['settlements_possible'] + access['settlements_bridge']
        return [('सडक सम्पर्क विच्छेद भएका बस्ती', f'**{nepali_int(access["settlements_high"])}**', nepali_int(any_count))]
    label = 'सडक पहुँच नभएका भवन' + (' (प्रभावित क्षेत्रबाहिर)' if access['has_footprint_class'] else '')
    return [
        (
            label,
            f'**{nepali_int(access["buildings_high"])}**' + units_ne(access['units_high']),
            nepali_int(access['buildings_any']) + units_ne(access['units_any']),
        ),
        (
            '… जसमध्ये नक्सामा गोरेटो पनि नभएका',
            nepali_int(access['air_high']),
            nepali_int(access['air_any']),
        ),
    ]


def key_figures(context):
    footprint, buildings, roads, bridges, health = (
        context[key] for key in ('footprint', 'buildings', 'roads', 'bridges', 'health')
    )
    rows = []
    if footprint:
        likely_km2, possible_km2 = footprint['likely_km2'], footprint['possible_km2']
        rows.append(
            (
                'बाढी / लेदो प्रभावित क्षेत्र',
                f'**{nepali_decimal(likely_km2)} वर्ग कि.मि.**',
                f'{nepali_decimal(likely_km2 + possible_km2)} वर्ग कि.मि.',
            )
        )
    else:
        rows.append(('बाढी / लेदो प्रभावित क्षेत्र', NOT_COMPUTED_NE, ''))
    if buildings:
        rows.append(
            (
                'प्रभावित क्षेत्रभित्र परेका भवन (अनुमानित, प्रमाणित क्षति होइन)',
                f'**{nepali_int(buildings["likely"])}** (नक्साङ्कित {nepali_int(buildings["total"])} मध्ये)',
                nepali_int(buildings['likely'] + buildings['possible']),
            )
        )
    else:
        rows.append(('प्रभावित क्षेत्रभित्र परेका भवन (अनुमानित, प्रमाणित क्षति होइन)', NOT_COMPUTED_NE, ''))
    if roads:
        rows += [
            (
                'मोटरबाटो अवरुद्ध हुने उच्च सम्भावना',
                f'**{nepali_km(roads["vehicle_blocked_km"])} कि.मि.** '
                f'({nepali_km(roads["vehicle_total_km"])} कि.मि. मध्ये)',
                nepali_km(roads['vehicle_blocked_km'] + roads['vehicle_possible_km']) + ' कि.मि.',
            ),
            (
                'प्रभावित पैदल बाटो / गोरेटो',
                f'{nepali_km(roads["path_blocked_km"])} कि.मि. ({nepali_km(roads["path_total_km"])} कि.मि. मध्ये)',
                nepali_km(roads['path_blocked_km'] + roads['path_possible_km']) + ' कि.मि.',
            ),
        ]
    else:
        rows.append(('सडक अवरुद्ध', NOT_COMPUTED_NE, ''))
    if bridges:
        rows.append(
            (
                'क्षतिको प्रमाण भएका पुल: मोटरबाटो · पैदल/झोलुङ्गे',
                f'**{nepali_int(bridges["vehicle_blocked"])}** ({nepali_int(bridges["vehicle_total"])} मध्ये) · '
                f'**{nepali_int(bridges["foot_blocked"])}** ({nepali_int(bridges["foot_total"])} मध्ये)',
                f'{nepali_int(bridges["vehicle_blocked"] + bridges["vehicle_possible"])} · '
                f'{nepali_int(bridges["foot_blocked"] + bridges["foot_possible"])}',
            )
        )
    if health:
        rows.append(
            (
                'प्रभावित क्षेत्रभित्र स्वास्थ्य संस्था',
                f'**{nepali_int(health["in_footprint"])}** ({nepali_int(health["total"])} मध्ये)',
                nepali_int(health['in_footprint'] + health['possibly_in_footprint']),
            )
        )
    rows += access_rows(context['access'])
    return ['## मुख्य तथ्याङ्क', '| | उच्च सम्भावना | बढीमा (सम्भावित समेत) |', '|---|---|---|'] + [
        f'| {label} | {likely} | {any_value} |' for label, likely, any_value in rows
    ]


def unit_label_ne(unit):
    return label_ne(unit['label'], unit.get('place'), unit.get('near'))


def key_messages(context):
    access, health = context['access'], context['health']
    messages = []
    if context['hardest']:
        messages.append(
            '**प्रभावित क्षेत्रभित्रका भवन:** '
            + ', '.join(f'{unit_label_ne(unit)} ({nepali_int(unit["n_fp"])})' for unit in context['hardest'][:4])
            + '।'
        )
    if context['repairs']:
        messages.append(
            '**सडक मर्मतका सम्भावित खण्ड:** '
            + ' '.join(f'#{devanagari(repair["rank"])} {repair_ne(repair)}।' for repair in context['repairs'][:3])
        )
    if health and (health['names_in_footprint'] or health['names_cut_off']):
        parts = []
        if health['names_in_footprint']:
            parts.append(
                'प्रभावित क्षेत्रभित्र: ' + ', '.join(local_name(name) for name in health['names_in_footprint'][:4])
            )
        if health['names_cut_off']:
            parts.append('सडक सम्पर्क विच्छेद: ' + ', '.join(local_name(name) for name in health['names_cut_off'][:4]))
        messages.append('**स्वास्थ्य संस्था:** ' + '; '.join(parts) + '।')
    if access and access['lost_exit']:
        names = access['lost_exit_names']
        examples = f' (जस्तै {", ".join(local_name(name) for name in names[:3])})' if names else ''
        messages.append(
            f'**{nepali_int(access["lost_exit"])} बस्ती** अझै स्थानीय सहर वा स्वास्थ्य संस्थासम्म पुग्छन् तर क्षेत्रबाहिर '
            f'जाने मुख्य सडक सम्पर्क गुमाएका छन्{examples}।'
        )
    if not messages:
        return []
    return ['## प्रभावित बस्ती'] + ['- ' + message for message in messages]


def priority_table(context):
    access = context['access']
    rows = context['priority'][:PRIORITY_ROWS]
    if not access or not rows:
        return []
    in_footprint = ' (+ प्रभावित क्षेत्रभित्र)' if access['has_footprint_class'] else ' (प्रभावित क्षेत्रभित्र)'
    lines = [
        f'## प्राथमिकता: सडक पहुँच नभएका बस्ती (भवन सङ्ख्याका आधारमा शीर्ष {devanagari(PRIORITY_ROWS)})',
        f'| # | बस्ती (क्षेत्र) | सडक पहुँच नभएका भवन{in_footprint} | हालको पहुँच | सडकबाट पैदल | प्रमाण |',
        '|---|---|---|---|---|---|',
    ]
    for rank, unit in enumerate(rows, 1):
        area = area_ne(unit.get('muni'), unit.get('ward'))
        name = table_cell(unit_label_ne(unit)) + (f' ({table_cell(area)})' if area else '')
        evidence = CONFIDENCE_NE.get(unit.get('conf'), '—') + (' †' if unit.get('spof') else '')
        lines.append(
            f'| {devanagari(rank)} | {name} | {nepali_int(unit["n_noroad"])} ({nepali_int(unit["n_fp"])}) | '
            f'{ACCESS_NE.get(unit.get("access"), "—")} | {table_cell(walk_ne(unit))} | {evidence} |'
        )
    more = max(0, int(context['n_priority'] or 0) - len(rows))
    footnote = f'थप {nepali_int(more)} ड्यासबोर्ड (Priorities) र priorities.csv मा। ' if more else ''
    if any(unit.get('spof') for unit in rows):
        footnote += '† एउटै अनिश्चित अवरोधले पहुँच निर्धारण गर्छ: पहिले त्यो क्रसिङ जाँच गर्नुहोस्। '
    lines.append(f'*{footnote}पैदल समय: उचाइ नक्साबाट अनुमान।*')
    return lines


def damage_lines(context):
    """Write the damage counts without the road-class detail to fit one A4 page."""
    buildings, roads, bridges, health = (context[key] for key in ('buildings', 'roads', 'bridges', 'health'))
    lines = ['## क्षति (घटनाअघिको OSM; नक्साङ्कित मध्ये प्रभावित)']
    if buildings:
        lines.append(
            f'- भवन: प्रभावित क्षेत्रभित्र **{nepali_int(buildings["likely"])} उच्च सम्भावना**, '
            f'{nepali_int(buildings["possible"])} सम्भावित, {nepali_int(buildings["unobserved"])} अवलोकन गर्न नसकिएको; '
            f'नक्साङ्कित {nepali_int(buildings["total"])} मध्ये (अनुमानित, प्रमाणित क्षति होइन)।'
        )
    else:
        lines.append(f'- भवन: {NOT_COMPUTED_NE}।')
    if not roads:
        lines.append(f'- सडक र पुल: {NOT_COMPUTED_NE}।')
    if bridges:
        lines.append(
            '- पुल (OSM पुल-मार्गअनुसार गनिएको, पहुँच-भागका आधारमा): मोटरबाटो '
            f'{nepali_int(bridges["vehicle_blocked"])} उच्च सम्भावना + {nepali_int(bridges["vehicle_possible"])} सम्भावित '
            f'(कुल {nepali_int(bridges["vehicle_total"])}); पैदल/झोलुङ्गे {nepali_int(bridges["foot_blocked"])} उच्च '
            f'सम्भावना + {nepali_int(bridges["foot_possible"])} सम्भावित (कुल {nepali_int(bridges["foot_total"])})।'
        )
    if health:
        names = (
            f'; सम्पर्क विच्छेद: {", ".join(local_name(name) for name in health["names_cut_off"][:5])}'
            if health['names_cut_off']
            else ''
        )
        lines.append(
            f'- स्वास्थ्य संस्था (क्षेत्रमा {nepali_int(health["total"])}): प्रभावित क्षेत्रभित्र '
            f'{nepali_int(health["in_footprint"])}, सडक सम्पर्क विच्छेद {nepali_int(health["cut_off"])} '
            f'(+{nepali_int(health["possibly_cut_off"])} सम्भावित){names}।'
        )
    return lines


def cutoff_lines(access):
    """Write the access counts without the unnamed-place detail to fit one A4 page."""
    if not access:
        return []
    net = access['net']
    buffer_km = net.get('buffer_km')
    if buffer_km:
        extent = (
            f' (क्षेत्रभन्दा {devanagari(f"{buffer_km:g}")} कि.मि. बाहिरसम्मको सञ्जाल विश्लेषण गरिएको; '
            'क्षेत्रबाहिरका सडक खुला मानिएका)'
        )
    else:
        extent = ' (सञ्जाल क्षेत्रभित्र मात्र)'
    lines = [
        '## सडक सम्पर्क विच्छेद — विवरण',
        '- *सडक सम्पर्क विच्छेद* = बस्तीबाट कुनै पनि सहर वा अस्पताल, वा विश्लेषित सडक सञ्जालबाट बाहिर जाने मुख्य सडकसम्म सडक '
        f'मार्ग नहुनु{extent}: {nepali_int(net.get("n_hospitals", 0))} अस्पताल, {nepali_int(net.get("n_towns", 0))} सहर, '
        f'{nepali_int(net.get("n_exits", 0))} मुख्य सडक निकास; {ROAD_ACCESS_RULE_NE}।',
    ]
    trail = f' (त्यहाँका {nepali_int(access["trail_cut"])} भवनको गोरेटो सम्पर्क पनि टुटेको)' if access['trail_cut'] else ''
    rest = (
        f'{nepali_int(access["settlements_detour"])} लामो घुमाउरो बाटो; '
        f'{nepali_int(access["settlements_no_road_before"])} मा घटनाअघि नै सडक थिएन{trail}।'
    )
    units = access['named_units']
    if units is not None:
        partial = f'; थप {nepali_int(access["partial"])} आंशिक रूपमा' if access['partial'] else ''
        lines.append(
            '- सडक पहुँच गुमेका नाम भएका बस्ती (बस्ती आफैं, वा प्रभावित क्षेत्रबाहिरका अधिकांश भवन): '
            f'**{nepali_int(units["high"])}** (बलियो प्रमाण), {nepali_int(units["possible"])} सम्भावित, '
            f'{nepali_int(units["bridge"])} पुल भत्किएमा{partial}; {rest}'
        )
        if access['footprint_buildings']:
            lines.append(
                f"- प्रभावित क्षेत्रभित्रका {nepali_int(access['footprint_buildings'])} भवन 'प्रभावित क्षेत्रभित्रका भवन' मा "
                'गनिएका छन्, सडक पहुँचको सङ्ख्यामा होइन।'
            )
    else:
        lines.append(
            f'- नाम भएका बस्ती: **{nepali_int(access["settlements_high"])}** को सडक सम्पर्क विच्छेद (बलियो प्रमाण), '
            f'{nepali_int(access["settlements_possible"])} सम्भावित, {nepali_int(access["settlements_bridge"])} पुल '
            f'भत्किएमा; {rest}'
        )
    return lines


def confidence_lines(context):
    footprint, access = context['footprint'], context['access']
    unobserved = f' ({nepali_decimal(footprint["unobserved_km2"])} वर्ग कि.मि. अवलोकन गर्न सकिएन)' if footprint else ''
    inferred_km2 = float(((footprint or {}).get('by_source_km2') or {}).get('continuity') or 0.0)
    inferred = ''
    if inferred_km2 >= 0.005:
        inferred = (
            f'उच्च सम्भावनाको {nepali_decimal(inferred_km2)} वर्ग कि.मि. क्षेत्र बादलले छेकेको छ। '
            'यो क्षेत्र सोही नदी खण्डको अनुमानित बाढी सतहबाट निकालिएको हो र सडक अवरोधको बलियो प्रमाणमा गनिँदैन। '
        )
    unverified = ''
    if access and access['unverified_buildings']:
        unverified = (
            f' **पहुँच अपुष्ट:** {nepali_int(access["unverified_buildings"])} भवन '
            f'({nepali_int(access["unverified_named"])} नाम भएका बस्ती) अवलोकन गर्न नसकिएका सडकबाट मात्र जोडिएका छन्।'
        )
    s1_text, s2_text = context['inputs']
    return [
        '## अवलोकन र सीमा',
        '- नदी नजिकको होचो भूभागमा राडार परिवर्तन, अप्टिकल परिवर्तन र पानीको अनुमानबाट प्रभावित क्षेत्र निकालिएको छ। '
        'पहाडी भूभागमा नदी च्यानलसँग जोडिएका परिवर्तन मात्र राखिन्छन्; कमजोर वा एकल स्रोतका परिवर्तन सम्भावित वर्गमा पर्छन्। '
        + inferred
        + ROAD_BREAK_RULE_NE
        + ' '
        + STRONG_EVIDENCE_RULE_NE,
        f'- देखिँदैन: {date_ne(context["newest_date"])} पछिका मर्मत वा नयाँ क्षति; बादल वा राडार छायाँमुनिको भूभाग{unobserved}; '
        'OSM मा नभएका भवन र सडक। सडक/पुलको अवस्था नक्साबाट अनुमान गरिएको हो (पुल पहुँच-भागका आधारमा), प्रत्यक्ष अवलोकन होइन।'
        + unverified,
        '',
        f'<sub>S1: {inputs_ne(s1_text)} · S2: {inputs_ne(s2_text)} · OSM (घटनाअघिको): '
        f'{context["osm_snapshot"]}</sub>',
        f'<sub>{context["attribution"]}</sub>',
    ]


def markdown(context):
    """Nepali sitrep from report.report_context(...), with the numbers in the order of the English page."""
    lines = header_lines(context) + notes(context['availability'])
    lines += key_figures(context)
    lines += key_messages(context)
    lines += priority_table(context)
    lines += damage_lines(context)
    lines += cutoff_lines(context['access'])
    lines += confidence_lines(context)
    return '\n'.join(lines)
