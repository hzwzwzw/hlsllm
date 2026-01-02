/* Inference for Llama-2 Transformer model in pure C, CPU verification mode. */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string>
#include <iostream>
#include <cstring>
#include <fcntl.h>

// 包含你的类型定义
#include "typedefs.h"
#include "config.h"

// 确保 forward.h 中包含 extern "C" void forward(...) 的声明
#include "forward.h" 

#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------
// Globals

void softmax(float *x, int size)
{
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++)
  {
    if (x[i] > max_val)
    {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++)
  {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++)
  {
    x[i] /= sum;
  }
}

template <int SIZE>
/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
void init_quantized_tensors(void **ptr, QuantizedTensor<SIZE> *tensor, int n, int size_each)
{
  void *p = *ptr;
  for (int i = 0; i < n; i++)
  {
    /* map quantized int8 values*/
    std::memcpy(tensor[i].q, p, size_each * sizeof(int8_t));
    p = (int8_t *)p + size_each;
    /* map scale factors */
    std::memcpy(tensor[i].s, p, (size_each / GS) * sizeof(float));

    p = (float *)p + size_each / GS;
  }
  *ptr = p; // advance ptr to current position
}

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void memory_map_weights(TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *w, void *ptr, uint8_t shared_classifier)
{
  int head_size = dim / n_heads;
  // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
  float *fptr = (float *)ptr; // cast our pointer to float*
  std::memcpy(w->rms_att_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_ffn_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_final_weight, fptr, dim * sizeof(float));
  fptr += dim;

  // now read all the quantized weights
  ptr = (void *)fptr; // now cast the pointer back to void*
  init_quantized_tensors(&ptr, w->q_tokens, 1, vocab_size * dim);
  // dequantize token embedding table
  dequantize<vocab_size * dim>(w->q_tokens, w->token_embedding_table, GS);

  init_quantized_tensors(&ptr, w->wq, n_layers, dim * (n_heads * head_size));
  init_quantized_tensors(&ptr, w->wk, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wv, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wo, n_layers, (n_heads * head_size) * dim);

  init_quantized_tensors(&ptr, w->w1, n_layers, dim * hidden_dim);
  init_quantized_tensors(&ptr, w->w2, n_layers, hidden_dim * dim);
  init_quantized_tensors(&ptr, w->w3, n_layers, dim * hidden_dim);

  if (shared_classifier)
  {
    std::memcpy(w->wcls, w->q_tokens, sizeof(QuantizedTensor<vocab_size * dim>));
  }
  else
  {
    init_quantized_tensors(&ptr, w->wcls, 1, dim * vocab_size);
  }
}

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void read_checkpoint(std::string checkpoint, Config *config, TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *weights)
{
  FILE *file = fopen(checkpoint.c_str(), "rb");
  if (!file)
  {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint.c_str());
    exit(EXIT_FAILURE);
  }
  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
  if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(EXIT_FAILURE); }
  int version;
  if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
  if (version != 2) { fprintf(stderr, "Bad version %d, need version 2\n", version); exit(EXIT_FAILURE); }
  int header_size = 256; 
  if (fread(config, sizeof(Config) - sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
  uint8_t shared_classifier; 
  if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
  int group_size; 
  if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
  config->GS = GS;
  
  fseek(file, 0, SEEK_END);     
  auto file_size = ftell(file); 
  fclose(file);
  
  auto fd = open(checkpoint.c_str(), O_RDONLY); 
  if (fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
  auto data = (float *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
  void *weights_ptr = ((char *)data) + header_size; 
  memory_map_weights(weights, weights_ptr, shared_classifier);
  close(fd);
  if (data != MAP_FAILED) { munmap(data, file_size); }
}

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void build_transformer(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *t, std::string checkpoint_path)
{
  read_checkpoint(checkpoint_path, &t->config, &t->weights);
}

// ----------------------------------------------------------------------------
// Tokenizer & Sampler (Unchanged Logic, condensed for brevity)
// ----------------------------------------------------------------------------

typedef struct { char *str; int id; } TokenIndex;
typedef struct { char **vocab; float *vocab_scores; TokenIndex *sorted_vocab; int vocab_size; unsigned int max_token_length; unsigned char byte_pieces[512]; } Tokenizer;

int compare_tokens(const void *a, const void *b) { return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str); }

void build_tokenizer(Tokenizer *t, std::string tokenizer_path, int vocab_size) {
  t->vocab_size = vocab_size;
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; 
  for (int i = 0; i < 256; i++) { t->byte_pieces[i * 2] = (unsigned char)i; t->byte_pieces[i * 2 + 1] = '\0'; }
  FILE *file = fopen(tokenizer_path.c_str(), "rb");
  if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path.c_str()); exit(EXIT_FAILURE); }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (fread(&len, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) { exit(EXIT_FAILURE); }
    t->vocab[i][len] = '\0'; 
  }
  fclose(file);
}

void free_tokenizer(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
  free(t->vocab); free(t->vocab_scores); free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
  char *piece = t->vocab[token];
  if (prev_token == 1 && piece[0] == ' ') { piece++; }
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) { piece = (char *)t->byte_pieces + byte_val * 2; }
  return piece;
}

void safe_printf(char *piece) {
  if (piece == NULL || piece[0] == '\0') { return; }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) { return; }
  }
  printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  TokenIndex tok = {.str = str}; 
  TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  if (text == NULL) { exit(EXIT_FAILURE); }
  if (t->sorted_vocab == NULL) {
    t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) { t->sorted_vocab[i].str = t->vocab[i]; t->sorted_vocab[i].id = i; }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }
  char *str_buffer = (char *)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
  size_t str_len = 0;
  *n_tokens = 0;
  if (bos) tokens[(*n_tokens)++] = 1;
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }
  for (char *c = text; *c != '\0'; c++) {
    if ((*c & 0xC0) != 0x80) { str_len = 0; }
    str_buffer[str_len++] = *c; str_buffer[str_len] = '\0';
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) { continue; }
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
    if (id != -1) { tokens[(*n_tokens)++] = id; }
    else { for (int i = 0; i < str_len; i++) { tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3; } }
    str_len = 0; 
  }
  while (1) {
    float best_score = -1e10; int best_id = -1; int best_idx = -1;
    for (int i = 0; i < (*n_tokens - 1); i++) {
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) { best_score = t->vocab_scores[id]; best_id = id; best_idx = i; }
    }
    if (best_idx == -1) { break; }
    tokens[best_idx] = best_id;
    for (int i = best_idx + 1; i < (*n_tokens - 1); i++) { tokens[i] = tokens[i + 1]; }
    (*n_tokens)--; 
  }
  if (eos) tokens[(*n_tokens)++] = 2;
  free(str_buffer);
}

typedef struct { float prob; int index; } ProbIndex;
typedef struct { int vocab_size; ProbIndex *probindex; float temperature; float topp; unsigned long long rng_state; } Sampler;

int sample_argmax(float *probabilities, int n) {
  int max_i = 0; float max_p = probabilities[0];
  for (int i = 1; i < n; i++) { if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; } }
  return max_i;
}
int sample_mult(float *probabilities, int n, float coin) {
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) { cdf += probabilities[i]; if (coin < cdf) { return i; } }
  return n - 1; 
}
int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a; ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob) return -1; if (a_->prob < b_->prob) return 1; return 0;
}
int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
  int n0 = 0; const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) { if (probabilities[i] >= cutoff) { probindex[n0].index = i; probindex[n0].prob = probabilities[i]; n0++; } }
  qsort(probindex, n0, sizeof(ProbIndex), compare);
  float cumulative_prob = 0.0f; int last_idx = n0 - 1; 
  for (int i = 0; i < n0; i++) { cumulative_prob += probindex[i].prob; if (cumulative_prob > topp) { last_idx = i; break; } }
  float r = coin * cumulative_prob; float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) { cdf += probindex[i].prob; if (r < cdf) { return probindex[i].index; } }
  return probindex[last_idx].index; 
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size; sampler->temperature = temperature; sampler->topp = topp; sampler->rng_state = rng_seed;
  sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}
void free_sampler(Sampler *sampler) { free(sampler->probindex); }
unsigned int random_u32(unsigned long long *state) { *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27; return (*state * 0x2545F4914F6CDD1Dull) >> 32; }
float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }
int sample(Sampler *sampler, float *logits) {
  int next;
  if (sampler->temperature == 0.0f) { next = sample_argmax(logits, sampler->vocab_size); }
  else {
    for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
    softmax(logits, sampler->vocab_size);
    float coin = random_f32(&sampler->rng_state);
    if (sampler->topp <= 0 || sampler->topp >= 1) { next = sample_mult(logits, sampler->vocab_size, coin); }
    else { next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin); }
  }
  return next;
}
long time_in_ms() { struct timespec time; clock_gettime(CLOCK_REALTIME, &time); return time.tv_sec * 1000 + time.tv_nsec / 1000000; }

// ----------------------------------------------------------------------------
// Generation Loop (CPU Version)
// ----------------------------------------------------------------------------

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void generate(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps)
{
  char *empty_prompt = "";
  if (prompt == NULL) { prompt = empty_prompt; }

  // encode the prompt
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) { fprintf(stderr, "something is wrong, expected at least 1 prompt token\n"); exit(EXIT_FAILURE); }

  std::cout << "Running pure CPU verification..." << std::endl;
  std::cout << "Transformer size: " << sizeof(*transformer) << " bytes" << std::endl;

  // 1. Allocate Memory for KV Cache and Outputs directly on Heap (no XRT BO)
  // Note: (dim * n_kv_heads) / n_heads  equals the size of KV vector per token per layer
  int kv_dim = (dim * n_kv_heads) / n_heads;
  size_t cache_size_floats = (size_t)n_layers * seq_len * kv_dim;
  
  std::cout << "Allocating KV Cache (" << (cache_size_floats * sizeof(float) * 2) / (1024*1024) << " MB)..." << std::endl;

  float *key_cache = (float *)calloc(cache_size_floats, sizeof(float));
  float *value_cache = (float *)calloc(cache_size_floats, sizeof(float));
  float *logits_out = (float *)malloc(vocab_size * sizeof(float));

  if (!key_cache || !value_cache || !logits_out) {
      fprintf(stderr, "Malloc failed for caches or logits!\n");
      exit(EXIT_FAILURE);
  }

  // start the main loop
  long start = 0; 
  int next;                     
  int token = prompt_tokens[0]; 
  int pos = 0;                  

  while (pos < steps)
  {
    // ------------------------------------------------------------------------
    // Call the HLS equivalent C++ function directly
    // ------------------------------------------------------------------------
    // extern "C" void forward(Transformer *t, int token, int pos, float *k, float *v, float *out)
    
    forward(transformer, token, pos, key_cache, value_cache, logits_out);
    
    // ------------------------------------------------------------------------

    // advance the state state machine
    if (pos < num_prompt_tokens - 1)
    {
      next = prompt_tokens[pos + 1];
    }
    else
    {
      // sample the next token from the logits
      next = sample(sampler, logits_out);
    }
    pos++;

    if (start == 0) start = time_in_ms(); // start timing after first token (prefill usually ignored in token/s for simplicity here, or start earlier)

    // data-dependent terminating condition
    if (next == 1) { break; }

    char *piece = decode(tokenizer, token, next);
    safe_printf(piece); 
    fflush(stdout);
    token = next;
  }
  printf("\n");

  if (pos > 1)
  {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
  }

  // Cleanup
  free(prompt_tokens);
  free(key_cache);
  free(value_cache);
  free(logits_out);
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
#ifndef TESTING

void error_usage()
{
  fprintf(stderr, "Usage:   run_cpu <checkpoint> [options]\n");
  fprintf(stderr, "Example: run_cpu model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate, default: generate\n");
  // removed -k kernelpath since we are running on CPU
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
  std::cout << "start CPU verification" << std::endl;
  std::string checkpoint_path = ""; 
  std::string tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;        
  float topp = 0.9f;               
  int steps = 256;                 
  char *prompt = NULL;             
  unsigned long long rng_seed = 0; 
  const char *mode = "generate";   

  if (argc >= 2) { checkpoint_path = argv[1]; }
  else { error_usage(); }

  for (int i = 2; i < argc; i += 2)
  {
    if (i + 1 >= argc) { error_usage(); } 
    if (argv[i][0] != '-') { error_usage(); } 
    if (strlen(argv[i]) != 2) { error_usage(); } 
    
    if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
    else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
    else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
    else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
    else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
    else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
    else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
    // ignore -k or other flags
  }

  if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0) temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0) steps = 0;

  static Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; 

  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  if (strcmp(mode, "generate") == 0)
  {
    // removed kernelpath argument
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
  }
  else
  {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  return 0;
}
#endif