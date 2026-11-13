export type SseFrame = { event: string; data: string };

/** Incremental SSE frame parser with a bounded unterminated-frame buffer. */
export class SseParser {
	private pending = '';
	private trailingCR = false;
	private readonly maxBuffer: number;
	private readonly onFrame: (frame: SseFrame) => void;

	constructor(onFrame: (frame: SseFrame) => void, maxBuffer = 1_048_576) {
		if (!Number.isSafeInteger(maxBuffer) || maxBuffer < 1) throw new RangeError('SSE buffer limit must be a positive safe integer.');
		this.onFrame = onFrame;
		this.maxBuffer = maxBuffer;
	}

	push(chunk: string): void {
		this.pending += this.normalizeNewlines(chunk, false);
		this.consume(false);
	}

	finish(): void {
		this.pending += this.normalizeNewlines('', true);
		this.consume(true);
	}

	private normalizeNewlines(chunk: string, eof: boolean): string {
		let text = chunk;
		if (this.trailingCR) {
			if (text.startsWith('\n')) text = text.slice(1);
			text = `\n${text}`;
			this.trailingCR = false;
		}
		if (!eof && text.endsWith('\r')) {
			text = text.slice(0, -1);
			this.trailingCR = true;
		}
		return text.replace(/\r\n|\r|\n/g, '\n');
	}

	private consume(eof: boolean): void {
		let separator = /\n\n/;
		let match: RegExpExecArray | null;
		while ((match = separator.exec(this.pending))) {
			if (match.index > this.maxBuffer) throw new Error('Market stream frame exceeded its size limit.');
			this.dispatch(this.pending.slice(0, match.index));
			this.pending = this.pending.slice(match.index + match[0].length);
		}
		if (eof && this.pending.length) {
			this.dispatch(this.pending);
			this.pending = '';
		} else if (this.pending.length > this.maxBuffer) {
			throw new Error('Market stream frame exceeded its size limit.');
		}
	}

	private dispatch(frame: string): void {
		let event = 'message';
		const data: string[] = [];
		for (const line of frame.split('\n')) {
			if (!line || line.startsWith(':')) continue;
			const colon = line.indexOf(':');
			const field = colon < 0 ? line : line.slice(0, colon);
			let value = colon < 0 ? '' : line.slice(colon + 1);
			if (value.startsWith(' ')) value = value.slice(1);
			if (field === 'event') event = value;
			else if (field === 'data') data.push(value);
		}
		if (data.length) this.onFrame({ event, data: data.join('\n') });
	}
}
