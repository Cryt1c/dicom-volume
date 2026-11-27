//! Temporary diagnostic: group the DICOM files in `dicom/` by identifying tags.
use dicom::object::open_file;
use dicom_dictionary_std::tags;
use std::collections::BTreeMap;
use std::fs;

fn get_str(
    obj: &dicom::object::FileDicomObject<dicom::object::InMemDicomObject>,
    tag: dicom::core::Tag,
) -> String {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn get_u32(
    obj: &dicom::object::FileDicomObject<dicom::object::InMemDicomObject>,
    tag: dicom::core::Tag,
) -> String {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_int::<i64>().ok())
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn main() {
    let mut paths: Vec<_> = fs::read_dir("dicom")
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("dcm"))
        })
        .collect();
    paths.sort();

    #[derive(Default)]
    struct Group {
        count: usize,
        instances: Vec<i64>,
        positions: Vec<f64>,
        series_numbers: BTreeMap<String, usize>,
        descriptions: BTreeMap<String, usize>,
    }

    let mut groups: BTreeMap<String, Group> = BTreeMap::new();
    let mut studies: BTreeMap<String, usize> = BTreeMap::new();
    let mut accessions: BTreeMap<String, usize> = BTreeMap::new();

    for path in &paths {
        let obj = open_file(path).unwrap();
        let study = get_str(&obj, tags::STUDY_INSTANCE_UID);
        let series = get_str(&obj, tags::SERIES_INSTANCE_UID);
        let acquisition = get_u32(&obj, tags::ACQUISITION_NUMBER);
        let accession = get_str(&obj, tags::ACCESSION_NUMBER);
        let description = get_str(&obj, tags::SERIES_DESCRIPTION);
        let protocol = get_str(&obj, tags::PROTOCOL_NAME);
        let series_number = get_u32(&obj, tags::SERIES_NUMBER);

        *studies.entry(study.clone()).or_default() += 1;
        *accessions.entry(accession.clone()).or_default() += 1;

        let key = format!("study={study} series={series} acq={acquisition} acc={accession}");
        let group = groups.entry(key).or_default();
        group.count += 1;
        *group.series_numbers.entry(series_number).or_default() += 1;
        *group
            .descriptions
            .entry(format!("{description} / {protocol}"))
            .or_default() += 1;
        if let Some(v) = obj
            .element(tags::INSTANCE_NUMBER)
            .ok()
            .and_then(|e| e.to_int::<i64>().ok())
        {
            group.instances.push(v);
        }
        if let Ok(pos) = obj.element(tags::IMAGE_POSITION_PATIENT)
            && let Ok(vals) = pos.to_multi_float64()
            && let Some(z) = vals.get(2)
        {
            group.positions.push(*z);
        }
    }

    println!("files: {}", paths.len());
    println!("distinct StudyInstanceUIDs: {}", studies.len());
    for (k, v) in &studies {
        println!("  {k}: {v} files");
    }
    println!("distinct AccessionNumbers: {}", accessions.len());
    for (k, v) in &accessions {
        println!("  '{k}': {v} files");
    }

    println!(
        "\ndistinct (study, series, acquisition, accession) groups: {}",
        groups.len()
    );
    for (key, g) in &groups {
        let min_i = g.instances.iter().min();
        let max_i = g.instances.iter().max();
        let min_p = g.positions.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_p = g
            .positions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "\n{key}\n  files={} series_number={:?} desc={:?}",
            g.count, g.series_numbers, g.descriptions
        );
        println!(
            "  instance numbers: {:?}..{:?} ({} have tag)",
            min_i,
            max_i,
            g.instances.len()
        );
        println!(
            "  z positions: {min_p:.3}..{max_p:.3} ({} have tag)",
            g.positions.len()
        );
        let mut sorted = g.positions.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let duplicates = sorted
            .windows(2)
            .filter(|w| (w[0] - w[1]).abs() < 1e-6)
            .count();
        println!("  duplicate z positions: {duplicates}");
        let mut sorted_i = g.instances.clone();
        sorted_i.sort();
        let dup_i = sorted_i.windows(2).filter(|w| w[0] == w[1]).count();
        println!("  duplicate instance numbers: {dup_i}");
    }
}
