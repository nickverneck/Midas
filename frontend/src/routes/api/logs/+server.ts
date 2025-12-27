import { json } from '@sveltejs/kit';
import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

export const GET = async () => {
    const logPath = path.resolve('../runs_ga/ga_log.csv');

    if (!fs.existsSync(logPath)) {
        return json({ error: 'Log file not found' }, { status: 404 });
    }

    const csvData = fs.readFileSync(logPath, 'utf-8');
    const parsed = Papa.parse(csvData, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    });

    return json(parsed.data);
};
