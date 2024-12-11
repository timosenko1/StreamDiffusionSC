<!-- frontend/src/routes/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import Warning from '$lib/components/Warning.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import { mediaStreamActions, onFrameChangeStore } from '$lib/mediaStream';
  import { getPipelineValues, deboucedPipelineValues } from '$lib/store';
  import { writable } from 'svelte/store';

  let pipelineParams: Fields;
  let pipelineInfo: PipelineInfo;
  let pageContent: string = '';
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  let queueCheckerRunning: boolean = false;
  let warningMessage: string = '';

  // New Parameters
  let guidanceScale: number = 1.2;
  let noiseAmount: number = 0.0;
  let grayscale: boolean = false;
  let controlnetMode: string = 'none';

  onMount(() => {
    getSettings();
  });

  async function getSettings() {
    try {
      const settings = await fetch('/api/settings').then((r) => r.json());
      pipelineParams = settings.input_params.properties;
      pipelineInfo = settings.info.properties;
      isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
      maxQueueSize = settings.max_queue_size;
      pageContent = settings.page_content || '';

      // Set default values from settings
      guidanceScale = pipelineParams.guidance_scale.default;
      noiseAmount = pipelineParams.noise_amount.default;
      grayscale = pipelineParams.grayscale.default;
      controlnetMode = pipelineParams.controlnet_mode.default;

      toggleQueueChecker(true);
    } catch (error) {
      console.error('Failed to fetch settings:', error);
      warningMessage = 'Failed to load settings. Please try again later.';
    }
  }

  function toggleQueueChecker(start: boolean) {
    queueCheckerRunning = start && maxQueueSize > 0;
    if (start) {
      getQueueSize();
    }
  }

  async function getQueueSize() {
    if (!queueCheckerRunning) {
      return;
    }
    try {
      const data = await fetch('/api/queue').then((r) => r.json());
      currentQueueSize = data.queue_size;
    } catch (error) {
      console.error('Failed to fetch queue size:', error);
    }
    setTimeout(getQueueSize, 10000);
  }

  function getStreamData() {
    if (isImageMode) {
      return {
        ...getPipelineValues(),
        guidance_scale: guidanceScale,
        noise_amount: noiseAmount,
        grayscale: grayscale,
        controlnet_mode: controlnetMode,
        image: $onFrameChangeStore?.blob
      };
    } else {
      return {
        ...deboucedPipelineValues,
        guidance_scale: guidanceScale,
        noise_amount: noiseAmount,
        grayscale: grayscale,
        controlnet_mode: controlnetMode,
      };
    }
  }

  // Reactive statement to monitor LCM status
  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: if ($lcmLiveStatus === LCMLiveStatus.TIMEOUT) {
    warningMessage = 'Session timed out. Please try again.';
  }

  let disabled = false;

  async function toggleLcmLive() {
    try {
      if (!isLCMRunning) {
        if (isImageMode) {
          await mediaStreamActions.enumerateDevices();
          await mediaStreamActions.start();
        }
        disabled = true;
        await lcmLiveActions.start(getStreamData);
        disabled = false;
        toggleQueueChecker(false);
      } else {
        if (isImageMode) {
          mediaStreamActions.stop();
        }
        lcmLiveActions.stop();
        toggleQueueChecker(true);
      }
    } catch (e) {
      warningMessage = e instanceof Error ? e.message : 'An error occurred.';
      disabled = false;
      toggleQueueChecker(true);
    }
  }

  // Watch for changes and send updates
  $: if ($lcmLiveStatus === LCMLiveStatus.CONNECTED) {
    lcmLiveActions.update(getStreamData());
  }
</script>

<svelte:head>
  <script
    src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"
  ></script>
</svelte:head>

<main class="container mx-auto flex max-w-5xl flex-col gap-3 px-4 py-4">
  <Warning bind:message={warningMessage}></Warning>
  <article class="text-center">
    {#if pageContent}
      {@html pageContent}
    {/if}
    {#if maxQueueSize > 0}
      <p class="text-sm">
        There are <span id="queue_size" class="font-bold">{currentQueueSize}</span>
        user(s) sharing the same GPU, affecting real-time performance. Maximum queue size is {maxQueueSize}.
        <a
          href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
          target="_blank"
          class="text-blue-500 underline hover:no-underline">Duplicate</a
        > and run it on your own GPU.
      </p>
    {/if}
  </article>
  {#if pipelineParams}
    <article class="my-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
      {#if isImageMode}
        <div class="sm:col-start-1">
          <VideoInput
            width={Number(pipelineParams.width.default)}
            height={Number(pipelineParams.height.default)}
          ></VideoInput>
        </div>
      {/if}
      <div class={isImageMode ? 'sm:col-start-2' : 'col-span-2'}>
        <ImagePlayer />
      </div>
      <div class="sm:col-span-2">
        <!-- Guidance Scale Slider -->
        <div class="mb-4">
          <label for="guidance_scale" class="block text-sm font-medium text-gray-700">
            Guidance Scale: {guidanceScale}
          </label>
          <input
            type="range"
            id="guidance_scale"
            min="0"
            max="20"
            step="0.1"
            bind:value={guidanceScale}
            class="w-full"
          />
        </div>

        <!-- Noise Amount Slider -->
        <div class="mb-4">
          <label for="noise_amount" class="block text-sm font-medium text-gray-700">
            Noise Amount: {noiseAmount}
          </label>
          <input
            type="range"
            id="noise_amount"
            min="0"
            max="1"
            step="0.01"
            bind:value={noiseAmount}
            class="w-full"
          />
        </div>

        <!-- Grayscale Checkbox -->
        <div class="mb-4 flex items-center">
          <input
            type="checkbox"
            id="grayscale"
            bind:checked={grayscale}
            class="mr-2 h-4 w-4 text-indigo-600 border-gray-300 rounded"
          />
          <label for="grayscale" class="text-sm">Convert to Grayscale</label>
        </div>

        <!-- ControlNet Mode Dropdown -->
        <div class="mb-4">
          <label for="controlnet_mode" class="block text-sm font-medium text-gray-700">
            ControlNet Mode
          </label>
          <select
            id="controlnet_mode"
            bind:value={controlnetMode}
            class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
          >
            <option value="none">None</option>
            <option value="canny">Canny</option>
            <option value="depth">Depth</option>
          </select>
        </div>

        <!-- Start/Stop Button -->
        <Button on:click={toggleLcmLive} {disabled} classList={'text-lg my-1 p-2'}>
          {#if isLCMRunning}
            Stop
          {:else}
            Start
          {/if}
        </Button>
        <PipelineOptions {pipelineParams}></PipelineOptions>
      </div>
    </article>
  {:else}
    <!-- loading -->
    <div class="flex items-center justify-center gap-3 py-48 text-2xl">
      <Spinner classList={'animate-spin opacity-50'}></Spinner>
      <p>Loading...</p>
    </div>
  {/if}
</main>

<style lang="postcss">
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }

  /* Additional styling for controls */
  input[type='range'] {
    appearance: none;
    height: 4px;
    background: #d1d5db;
    border-radius: 2px;
    outline: none;
  }

  input[type='range']::-webkit-slider-thumb {
    appearance: none;
    width: 16px;
    height: 16px;
    background: #3b82f6;
    border-radius: 50%;
    cursor: pointer;
    transition: background 0.3s;
  }

  input[type='range']::-moz-range-thumb {
    width: 16px;
    height: 16px;
    background: #3b82f6;
    border-radius: 50%;
    cursor: pointer;
    transition: background 0.3s;
  }

  input[type='range']:hover::-webkit-slider-thumb {
    background: #2563eb;
  }

  input[type='range']:hover::-moz-range-thumb {
    background: #2563eb;
  }
</style>